local LSTM = {}

function LSTM.lstm(num_inputs, num_actions, num_hidden_layer_neurons, num_hidden_layers, num_outputs, dropout)
  dropout = dropout or 0 

  -- there will be 2 * num_hidden_layers + 1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,num_hidden_layers do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,num_hidden_layers do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then 
       -- convert class lable and action to 1-of-k representations
      local p = nn.Parallel(3, 2)
      p:add(OneHot(num_inputs))
      p:add(OneHot(num_actions))
      x = p(inputs[1])
      input_size_L = num_inputs + num_actions
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = num_hidden_layer_neurons
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * num_hidden_layer_neurons)(x)
    local h2h = nn.Linear(num_hidden_layer_neurons, 4 * num_hidden_layer_neurons)(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, h2h})
    -- decode the gates
    local sigmoid_chunk = nn.Narrow(2, 1, 3 * num_hidden_layer_neurons)(all_input_sums)
    sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
    local in_gate = nn.Narrow(2, 1, num_hidden_layer_neurons)(sigmoid_chunk)
    local forget_gate = nn.Narrow(2, num_hidden_layer_neurons + 1, num_hidden_layer_neurons)(sigmoid_chunk)
    local out_gate = nn.Narrow(2, 2 * num_hidden_layer_neurons + 1, num_hidden_layer_neurons)(sigmoid_chunk)
    -- decode the write inputs
    local in_transform = nn.Narrow(2, 3 * num_hidden_layer_neurons + 1, num_hidden_layer_neurons)(all_input_sums)
    in_transform = nn.Tanh()(in_transform)
    -- perform the LSTM update
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    
    table.insert(outputs, next_c)
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

return LSTM
