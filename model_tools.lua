local model_tools = {}

function model_tools.save_model(file_name, options, weights, stats)
  print(string.format("Saving model to file '%s'", file_name))
  local f = torch.DiskFile(file_name, 'w')
  f:writeObject({ version = 0 })
  f:writeObject(options)
  f:writeObject(weights)
  f:writeObject(stats)
  f:close()
end


function model_tools.load_model(file_name)
  local f = torch.DiskFile(file_name, 'r')
  local header = f:readObject()
  local options = f:readObject()
  local weights = f:readObject()
  local stats = f:readObject()
  f:close()
  return options, weights, stats
end


function model_tools.clone_state(state, set_zero)
  local clone = {}
  for k,v in ipairs(state) do
    clone[k] = v:clone()
    if set_zero then clone[k]:zero() end
  end
  return clone
end


function model_tools.unroll_model(net, criterion, depth)
  local clones = { net={}, criterion={} }

  for i=1,depth do
    table.insert(clones.net, net:clone('weight', 'bias', 'gradWeight', 'gradBias'))
    table.insert(clones.criterion, criterion:clone())
  end

  return clones
end

function model_tools.format_size(size)
  return table.concat(torch.totable(size), ', ')
end

function model_tools.copy_missing_values(destination, source)
  for k,v in pairs(source) do
    if destination[k] == nil then
      destination[k] = v
    end
  end
end

return model_tools
