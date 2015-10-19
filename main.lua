require 'optim'
require 'nngraph'
require 'layers.OneHot'
local RNN = require 'models.RNN'
local LSTM = require 'models.LSTM'
local GRU = require 'models.GRU'

require 'agent'
require 'batchgenerator'
mt = require 'model_tools'


local ModelType = {
  RNN = 0,
  LSTM = 1,
  GRU = 2
}


local opt = {
  world_size = { 20, 20 },
  num_layers = 2,
  layer_size = 64,
  initial_learn_rate = 1e-3,
  rms_decay = 0.97,
  batch_seq_length = 50,   -- number of steps for truncated backprapagation through time,
  batch_size = 200,
  delta_clipping = 20,
  gpuid = 0,
  seed = 42,
  model_type = ModelType.LSTM,
  alphabet_size = 26,
  max_noturn_steps = 10,
  learn_rate_decay = 0.985,
  learn_rate_decay_interval = 500,
  snapshot_interval = 1000,
  enable_action_input = true,
  dropout = 0,
  model_basename = 'model'
}


local function create_world(world_size, alphabet_size)
  alphabet_size = alphabet_size or 26
  local world = torch.rand(world_size) * alphabet_size + 1
  return world:long():clamp(1, alphabet_size)
end


-- keep some statistics about the training
local training_stats = {}

local zero_hidden_state = {}
local rnn_hidden_state


-- create
local function create_optimization_target(unrolled, weights, gradient, batch_generator)

  local function loss_and_gradient(w)
    if w ~= weights then
      weights:copy(w)
    end
    gradient:zero()

    local x, y = batch_generator:next()

    if opt.gpuid >= 0 then -- ship the input arrays to GPU
      x = x:float():cuda()
      y = y:float():cuda()
    end

    local rnn_state = { [0] = rnn_hidden_state }
    local predictions = {}
    local loss = 0

    -- forward pass
    for t=1,opt.batch_seq_length do
      local net = unrolled.net[t]
      local criterion = unrolled.criterion[t]
      net:training()
      local output = net:forward({x[{{}, {t}, {}}], unpack(rnn_state[t-1])})

      -- add hidden state to rnn_state list
      local hidden_state = {}
      for i=1,#rnn_hidden_state do
        table.insert(hidden_state, output[i])
      end
      rnn_state[t] = hidden_state

      -- store predictions
      predictions[t] = output[#output]

      -- compute loss
      loss = loss + criterion:forward(predictions[t], y[{{}, t}])
    end
    loss = loss / #unrolled.net

    -- backward pass, start with zero gradients (net's output are the layers' hidden states + log softmax output)
    local deltas = { [opt.batch_seq_length] = mt.clone_state(zero_hidden_state) }
    for t=opt.batch_seq_length,1,-1 do
      local net = unrolled.net[t]
      local criterion = unrolled.criterion[t]

      local delta_output = criterion:backward(predictions[t], y[{{}, t}])
      table.insert(deltas[t], delta_output)

      -- clip delta values
      for i=1,#deltas[t] do
        deltas[t][i]:clamp(-opt.delta_clipping, opt.delta_clipping)
      end

      local delta_inputs = net:backward({x[{{}, {t}, {}}], unpack(rnn_state[t-1])}, deltas[t])
      deltas[t-1] = delta_inputs
      table.remove(deltas[t-1], 1)

    end

    return loss, gradient
  end

  return loss_and_gradient
end


-- note: conversion to CudaTensor has to be done before parameter sharing


local function init_gpu()
  -- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
  if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
      print('using CUDA on GPU ' .. opt.gpuid .. '...')
      cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
      cutorch.manualSeed(opt.seed)
    else
      print('Falling back on CPU mode')
      opt.gpuid = -1 -- overwrite user setting
    end
  end
end





local function parse_size(size_string)
  local s = string.split(size_string, ",")
  for k,v in pairs(s) do
    s[k] = tonumber(v)
  end
  return torch.LongStorage(s)
end


local function settings_string(t)
  local r = {}
  for k,v in pairs(t) do
    table.insert(r, string.format("%s: %s; ", k, tostring(v)))
  end
  return table.concat(r)
end



local function main()
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Matrix Walker')
  cmd:text()
  cmd:option('-w', 'world.txt', 'path to world definition file')
  cmd:option('-worldsize', '20,20', 'gridsize of world')
  cmd:option('-m', 'model.txt', 'model file name (weights etc.)')
  cmd:option('-gpuid', opt.gpuid, 'which gpu to use. -1 = use CPU')
  cmd:option('-l', opt.num_layers, 'number of layers')
  cmd:option('-n', opt.layer_size, 'number of neurons/cells per hidden layer')
  cmd:option('-model', 'LSTM', 'neural network to use (valid values: RNN, LSTM, GRU)')
  cmd:option('-dropout', 0, 'dropout probability (e.g. 0.5) for regularization, after each hidden layer. 0 = no dropout')
  cmd:option('-maxiter', 25000, 'max number of iterations')
  cmd:option('-maxnoturnsteps', opt.max_noturn_steps, 'maximum number of steps before changing direction (1 = vanilla random walk)')
  cmd:option('-noactions', true, 'disable action input')
  cmd:option('-seqlen', 50, 'per batch network unrolling/BPTT depth')
  cmd:option('-batchsize', 200, 'number of agents evaluated in parallel for one gradient step')
  cmd:text()

  -- parse command line arguments
  local params = cmd:parse(arg)

  opt.learn_rate = opt.initial_learn_rate
  opt.gpuid = params.gpuid
  opt.num_layers = params.l
  opt.layer_size = params.n
  opt.dropout = params.dropout
  opt.model_type = ModelType[string.upper(params.model)]
  opt.world_size = parse_size(params.worldsize)
  opt.enable_action_input = params.noactions
  opt.batch_seq_length = params.seqlen
  opt.batch_size = params.batchsize 
  if opt.max_noturn_steps > 0 then
    opt.max_noturn_steps = params.maxnoturnsteps
  end

  -- load existing or generate new world file
  local w
  local world_file_path = params.w
  if paths.filep(world_file_path) then
    local f = torch.DiskFile(world_file_path, 'r')
    w = f:readObject()
    print(string.format("Loaded existing world '%s' (size: %s).", world_file_path, mt.format_size(w:size())))
    f:close()
  end

  if not w then
    print("Creating new world...")
    w = create_world(opt.world_size, opt.alphabet_size)
    local f = torch.DiskFile(world_file_path, 'w')
    f:writeObject(w)
    f:close()
    print(string.format("Saved new world to file '%s' (size: %s).", world_file_path, mt.format_size(w:size())))
  end

  init_gpu()

  opt.model_basename = params.m
  local _weights
  if paths.filep(params.m) then
    local _opt = opt
    opt, _weights, training_stats = mt.load_model(params.m)
    mt.copy_missing_values(opt, _opt)
  end

  print("Effective options:")
  print(opt)

  -- create network
  local action_count = w:dim() * 2
  local net
  if opt.model_type == ModelType.RNN then
    print('RNN')
    net = RNN.rnn(opt.alphabet_size, action_count, opt.layer_size, opt.num_layers, opt.alphabet_size, opt.dropout)
  elseif opt.model_type == ModelType.LSTM then
    print('LSTM')
    net = LSTM.lstm(opt.alphabet_size, action_count, opt.layer_size, opt.num_layers, opt.alphabet_size, opt.dropout)
  elseif opt.model_type == ModelType.GRU then
    print('GRU')
    net = GRU.gru(opt.alphabet_size, action_count, opt.layer_size, opt.num_layers, opt.alphabet_size, opt.dropout)
  end
  local criterion = nn.ClassNLLCriterion()

  if opt.gpuid>= 0 then
    net:cuda()
    criterion:cuda()
  end

  -- initial hidden state
  for i=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.layer_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(zero_hidden_state, h_init:clone())
    if opt.model_type == ModelType.LSTM then
      table.insert(zero_hidden_state, h_init:clone()) -- second matrix required for LSTM cell states
    end
  end

  rnn_hidden_state = mt.clone_state(zero_hidden_state)

  -- flatten parameters
  local weights, gradient = net:getParameters()
  if not _weights then
    print('Initializing weights')
    weights:uniform(-0.05, 0.05) -- initialize weights
  else
    print('using existing weights')
    weights:copy(_weights)
  end

  local unrolled = mt.unroll_model(net, criterion, opt.batch_seq_length)
  print(string.format('Model parameters: %d', weights:nElement()))

  local bg = BatchGenerator.new(w, opt.batch_size, opt.batch_seq_length, opt.max_noturn_steps, opt.enable_action_input)
  print("BatchGenerator settings: " .. settings_string(bg:get_settings()))
  local optimization_target = create_optimization_target(unrolled, weights, gradient, bg)
  local rmsprop_state = { learningRate = opt.learn_rate, alpha = opt.rms_decay }

  local function format_snapshot_name(i, loss)
    return string.format('%s_i%06d_%.4f.t7', opt.model_basename, i, loss)
  end

  local iterations = params.maxiter
  for i=1,iterations do

    if #training_stats % opt.learn_rate_decay_interval == 0 then
      opt.learn_rate = opt.learn_rate * opt.learn_rate_decay
      rmsprop_state.learningRate = opt.learn_rate
    end

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(optimization_target, weights, rmsprop_state)
    loss = loss[1]
    local time = timer:time().real

    table.insert(training_stats, { loss = loss, time = time })
    print(string.format("Iteration %d (total %d), lr = %f, loss: %f, time: %fs", i, #training_stats, opt.learn_rate, loss, time))

    if i % opt.snapshot_interval == 0 then
      mt.save_model(format_snapshot_name(#training_stats, loss), opt, weights, training_stats)
    end

    if i % 25 == 0 then
      collectgarbage()
    end

  end
  
  mt.save_model(opt.model_basename .. ".t7", opt, weights, training_stats)

end

main()
