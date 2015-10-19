require 'nn'
require 'nngraph'
require 'optim'

local INPUT_SIZE = 3
local NEURONS_PER_LAYER = 5
local OUTPUT_NEURONS = 2

function create_xor_mlp(input_count, hidden_neuron_count, output_count)

  local input = nn.Identity()()  -- declare the neural network input node
  
  -- add a inner product (fully connected layer)
  local fc1 = nn.Linear(input_count, hidden_neuron_count)(input)
  
  -- squash the result
  local out1 = nn.Tanh()(fc1)
  
  -- add a second inner product layer
  local fc2 = nn.Linear(hidden_neuron_count, output_count)(out1)
  
  -- and pass it through the tanh nonlinearity
  local output = nn.Tanh()(fc2)
  
  -- create our MLP module
  local mlp = nn.gModule({ input }, { output })

  -- initialize parameters
  weights, weight_grads = mlp:getParameters()
  
  weights:uniform(-0.1, 0.1)
  
  return mlp
end  

-- create net
net = create_xor_mlp(2, 3, 1)


-- let's try to learn the xor function
local xor_batch_input = torch.Tensor({ { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } })
local xor_batch_target = torch.Tensor({ { 0 }, { 1 }, { 1 }, { 0 } })

local criterion = nn.MSECriterion()

local function perform_gradient_step(x, y, learn_rate)
  net:training() -- switch to training mode
  net:zeroGradParameters()
  local prediction = net:forward(x)
  local error = criterion:backward(prediction, y)
  net:backward(x, error)
  net:updateParameters(learn_rate)
end

local function evaluate_batch(x, y)
  net:evaluate() -- switich to evaluation mode
  local prediction = net:forward(x)
  return criterion:forward(prediction, y)
end

-- simple SGD
local function perform_naive_sgd() 
  for i=1,500 do
   perform_gradient_step(xor_batch_input, xor_batch_target, 0.5)
   local error = evaluate_batch(xor_batch_input, xor_batch_target)
   print(string.format("Epoch %d; Error: %f", i, error)) 
  end
end

local function create_optimizer(x, y)
  local weights, weight_derivatives = net:getParameters()
  
  local function calculate_gradient(w)
    if w ~= weigths then
      weights:copy(w)
    end
    net:training()
    net:zeroGradParameters()
    local prediction = net:forward(x)
    local error = criterion:backward(prediction, y)
    net:backward(x, error)
    return error:sum(), weight_derivatives
  end
  
  return calculate_gradient, weights
end
  

local function train_with_rmsprop()
  rmsprop_state = { learningRate = 0.01, alpha = 0.99 }
  local opfn, weights = create_optimizer(xor_batch_input, xor_batch_target)
  for i=1,80 do
    optim.rmsprop(opfn, weights, rmsprop_state)
    local error = evaluate_batch(xor_batch_input, xor_batch_target)
   print(string.format("Epoch %d; Error: %f", i, error)) 
  end
end

train_with_rmsprop()

