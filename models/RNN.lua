local RNN = {}

function RNN.rnn(num_inputs, num_actions, num_hidden_layer_neurons, num_hidden_layers, num_outputs, dropout)

  -- there are num_hidden_layers + 1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,num_hidden_layers do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
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

    -- RNN tick
    local i2h = nn.Linear(input_size_L, num_hidden_layer_neurons)(x)
    local h2h = nn.Linear(num_hidden_layer_neurons, num_hidden_layer_neurons)(prev_h)
    local next_h = nn.Tanh()(nn.CAddTable(){i2h, h2h})

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

return RNN