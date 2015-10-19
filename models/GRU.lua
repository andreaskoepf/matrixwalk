local GRU = {}

--[[
Creates one timestep of one GRU
Paper reference: http://arxiv.org/pdf/1412.3555v1.pdf
]]--
function GRU.gru(num_inputs, num_actions, num_hidden_layer_neurons, num_hidden_layers, num_outputs, dropout)
  
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,num_hidden_layers do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  function new_input_sum(insize, xv, hv)
    local i2h = nn.Linear(insize, num_hidden_layer_neurons)(xv)
    local h2h = nn.Linear(num_hidden_layer_neurons, num_hidden_layer_neurons)(hv)
    return nn.CAddTable()({i2h, h2h})
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,num_hidden_layers do

    local prev_h = inputs[L+1]
    if L == 1 then 
       -- convert class lable and action to 1-of-k representations
      local p = nn.Parallel(3, 2)
      p:add(OneHot(num_inputs))
      p:add(OneHot(num_actions))
      x = p(inputs[1])
      input_size_L = num_inputs + num_actions
    else 
      x = outputs[L-1]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = num_hidden_layer_neurons
    end

    -- GRU tick
    -- forward the update and reset gates
    local update_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
    local reset_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
    local p2 = nn.Linear(num_hidden_layer_neurons, num_hidden_layer_neurons)(gated_hidden)
    local p1 = nn.Linear(input_size_L, num_hidden_layer_neurons)(x)
    local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
    -- compute new interpolated hidden state, based on the update gate
    local zh = nn.CMulTable()({update_gate, hidden_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
    local next_h = nn.CAddTable()({zh, zhm1})

    table.insert(outputs, next_h)
  end

    -- add softmax output layer (decoder)
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(num_hidden_layer_neurons, num_outputs)(top_h)
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return GRU
