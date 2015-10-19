gfx = require 'gfx.js'
require 'cunn'
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
  initial_learn_rate = 2e-3,
  rms_decay = 0.97,
  batch_seq_length = 50,   -- number of steps for truncated backprapagation through time,
  batch_size = 200,
  delta_clipping = 10,
  gpuid = 0,
  seed = 42,
  model_type = ModelType.LSTM,
  alphabet_size = 26,
  max_noturn_steps = 10,
  learn_rate_decay = 0.975,
  learn_rate_decay_interval = 100,
  snapshot_interval = 100,
  enable_action_input = true,
  dropout = 0,
  model_basename = 'model'
}


local worlds_base_path = '/home/koepf/lua/work/results/worlds' 
local base_path_1 = '/home/koepf/lua/work/results/1'

local model1_4096_1d = { 'model_4096_1d_gru.txt', 'model_4096_1d_lstm.txt', 'model_4096_1d_rnn.txt' }
local model1_4096_2d = { 'model_4096_2d_gru.txt', 'model_4096_2d_lstm.txt', 'model_4096_2d_rnn.txt' }
local model1_4096_3d = { 'model_4096_3d_gru.txt', 'model_4096_3d_lstm.txt', 'model_4096_3d_rnn.txt' }
local model1_729_1d = { 'model_729_1d_gru.txt', 'model_729_1d_lstm.txt', 'model_729_1d_rnn.txt' }
local model1_729_2d = { 'model_729_2d_gru.txt', 'model_729_2d_lstm.txt', 'model_729_2d_rnn.txt' }
local model1_729_3d = { 'model_729_3d_gru.txt', 'model_729_3d_lstm.txt', 'model_729_3d_rnn.txt' }


local base_path_2 = '/home/koepf/lua/work/results/2'

local model2_729_1d = { 'model_729_1d_gru.t7', 'model_729_1d_lstm.t7', 'model_729_1d_rnn.t7' }
local model2_729_2d = { 'model_729_2d_gru.t7', 'model_729_2d_lstm.t7', 'model_729_2d_rnn.t7' }
local model2_729_3d = { 'model_729_3d_gru.t7', 'model_729_3d_lstm.t7', 'model_729_3d_rnn.t7' }

local model2_4096_1d = { 'model_4096_1d_gru.t7', 'model_4096_1d_rnn.t7', 'model_4096_1d_lstm_d02.t7', 'model_4096_1d_gru_d02.t7' }
local model2_4096_2d = { 'model_4096_2d_gru.t7', 'model_4096_2d_lstm.t7', 'model_4096_2d_rnn.t7' }
local model2_4096_3d = { 'model_4096_3d_gru.t7', 'model_4096_3d_lstm.t7', 'model_4096_3d_rnn.t7' }



local function load_training_history(fn)
  local f = torch.DiskFile(fn, 'r')
  local header = f:readObject()
  local options = f:readObject()
  local weights = f:readObject()
  local stats = f:readObject()
  f:close()
  return stats
end


local function thin_data(d, step)
  local r = {}
  local n = #d
  for i=1,#d,step do
    local l = d[i].loss
    if type(l) == 'table' then
      l = l[1]
    end
    table.insert(r, { x=i, y=l })
  end
  return r
end

-- create chart
local function plot_chart(models, base_path)
  local data = {}
  
  for i,m in ipairs(models) do
    local stats = load_training_history(paths.concat(base_path, m))
    local set = 
    {
      key = m,
    --  color = '#00f',
      values = thin_data(stats, 100)
    }
    table.insert(data, set)
  end
  
  gfx.chart(data, {
     chart = 'line', -- or: bar, stacked, multibar, scatter
     width = 800,
     height = 500
  })
end


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


local zero_hidden_state = {}
local rnn_hidden_state


local function evaluate_batch(unrolled, batch_generator)
  local x, y = batch_generator:next()
  
  if opt.gpuid >= 0 then -- ship the input arrays to GPU
    x = x:float():cuda()
    y = y:float():cuda()
  end
    
  local rnn_state = { [0] = rnn_hidden_state }
  local predictions = {}
  local loss = {}
  
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
    table.insert(loss, criterion:forward(predictions[t], y[{{}, t}]))
  end
  
  return loss, predictions, y
end


local function evaluate_model_predictions(world_file_path, model_file_path)

    -- load existing or generate new world file
  local w
  local f = torch.DiskFile(world_file_path, 'r')
  w = f:readObject()
  print(string.format("Loaded existing world '%s' (size: %s).", world_file_path, mt.format_size(w:size())))
  f:close()
  
  local _weights
  local training_stats
  local _opt = opt
  opt, _weights, training_stats = mt.load_model(model_file_path)
  mt.copy_missing_values(opt, _opt)
  
  opt.batch_size = 50
  opt.batch_seq_length = 50
  
  init_gpu()

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

  -- convert tensors to cuda tensors
  if opt.gpuid>= 0 then
    net:cuda()
    criterion:cuda()
  end

  zero_hidden_state = {}
  rnn_hidden_state = nil
  
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
  print('using existing weights')
  weights:copy(_weights)

  local unrolled = mt.unroll_model(net, criterion, opt.batch_seq_length)
  
  local bg = BatchGenerator.new(w, opt.batch_size, opt.batch_seq_length, opt.max_noturn_steps, opt.enable_action_input)
  
  local w = { }
  local i = 1
  for j=1,5 do
    bg:randomize_positions()
    local loss, predictions, targets = evaluate_batch(unrolled, bg)
    for j,l in ipairs(loss) do
      table.insert(w, { x=i, y=l})
      i = i + 1
    end
  end
  
  return w
end

local function plot_prediction_loss(world_file_name, model_names, base_path)
  local data = {}
  
  for i,m in ipairs(model_names) do

    local values = evaluate_model_predictions(paths.concat(worlds_base_path, world_file_name), paths.concat(base_path, m)) 
    
    local set = 
    {
      key = m,
    --  color = '#00f',
      values = values
    }
    table.insert(data, set)
  end

  gfx.chart(data, {
     chart = 'line', -- or: bar, stacked, multibar, scatter
     width = 800,
     height = 600
  })
end


-- experiment 1
--[[plot_chart(model1_4096_1d, base_path_1)
plot_chart(model1_4096_2d, base_path_1)
plot_chart(model1_4096_3d, base_path_1)
plot_chart(model1_729_1d, base_path_1)
plot_chart(model1_729_2d, base_path_1)
plot_chart(model1_729_3d, base_path_1)]]--

-- experiment 2
--[[plot_chart(model2_4096_1d, base_path_2)
plot_chart(model2_4096_2d, base_path_2)
plot_chart(model2_4096_3d, base_path_2)
plot_chart(model2_729_1d, base_path_2)
plot_chart(model2_729_2d, base_path_2)
plot_chart(model2_729_3d, base_path_2)]]--

plot_prediction_loss("world_4096_2d.txt", model2_4096_2d, base_path_2)
plot_prediction_loss("world_4096_3d.txt", model2_4096_3d, base_path_2)

plot_prediction_loss("world_729_2d.txt", model2_729_2d, base_path_2)
plot_prediction_loss("world_729_3d.txt", model2_729_3d, base_path_2)



                                      
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   
                                                                   


                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
                                                                                                                               
            
  