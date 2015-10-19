require 'agent'

local BatchGenerator = torch.class('BatchGenerator')

function BatchGenerator:__init(world, batch_size, sequence_length, max_noturn_steps, enable_action_input)
  self.world = world
  self.batch_size = batch_size  -- number of concurrent agents
  self.sequence_length = sequence_length  -- sequence length of one batch
  self.max_noturn_walk = max_noturn_steps or 1

  self.enable_action_input = enable_action_input
  if self.enable_action_input == nil then
    self.enable_action_input = true
  end
  self.agents = {}
  self.steps_until_turn = {}
  for i=1,batch_size do
    self.agents[i] = Agent.new(world)
    self.steps_until_turn[i] = 0
  end
end

function BatchGenerator:get_positions()
  local positions = {}
  for k,v in pairs(self.agents) do
    positions[k] = v.pos
  end
  return positions
end

function BatchGenerator:set_positions(position_list)
  for k,v in pairs(position_list) do
    if self.agents[k] then
      self.agents[k].pos = v
      self.steps_until_turn[k] = 0
    end
  end
end

function BatchGenerator:randomize_positions()
  for k,v in pairs(self.agents) do
    v:randomize_pos()
    self.steps_until_turn[k] = 0
  end
end

-- create a new table and fill it with selected values from the 'source' table.
local function extract(source, keys)
  local r = {}
  for i,v in ipairs(keys) do
    r[v] = source[v]
  end
  return r
end

function BatchGenerator:get_settings()
  return extract(self, { 'batch_size', 'sequence_length', 'max_noturn_walk', 'enable_action_input' })
end

function BatchGenerator:next()
  local x = torch.zeros(self.batch_size, self.sequence_length, 2)
  local y = torch.DoubleTensor(self.batch_size, self.sequence_length)
  for t = 1,self.sequence_length do
    for i = 1,#self.agents do
      local a = self.agents[i]
      x[{i, t, 1}] = a:sensor()         -- input 1: current observation

      local action = a.last_action
      self.steps_until_turn[i] = self.steps_until_turn[i] - 1
      if self.steps_until_turn[i] <= 0 then
        action = a:choose_action()
        self.steps_until_turn[i] = torch.random(self.max_noturn_walk)
      end

      -- input 2: executed action
      -- In order to use a single network topology we simply always pass action 1 if action input is disabled.
      x[{i, t, 2}] = self.enable_action_input and action or 1

      a:move(action)
      y[{i, t}] = a:sensor()            -- target: observation after action
    end
  end
  return x, y
end
  