
--[[
A simple agent that keeps track of it's position
and ensures that it stays inside the world boundaries.
]]--
local Agent = torch.class('Agent')

-- If initial_pos is omitted position set to a random value.
function Agent:__init(world, initial_pos)
  self.world = world
  self.d = world:dim()
  self.size = world:size()
  self.action_count = self.d * 2
  self.pos = initial_pos
  if not self.pos then
    self.pos = torch.LongStorage(self.d)
    self:randomize_pos()
  end
end

function Agent:randomize_pos()
  for i=1,self.d do
    self.pos[i] = torch.random() % self.size[i] + 1
  end
end

function Agent:choose_action()
  return torch.random() % self.action_count + 1 -- select axis and direction for agent step
end

function Agent:move(action)
  local d = math.ceil(action / 2)
  if (action % 2) == 0 then  -- increment on even numbers, decrement otherwise
    self.pos[d] = self.pos[d] + 1
    if (self.pos[d] > self.size[d]) then
      self.pos[d] = 1
    end
  else
    self.pos[d] = self.pos[d] - 1
    if self.pos[d] < 1 then
      self.pos[d] = self.size[d]
    end
  end
  self.last_action = action
end

function Agent:sensor()
  return self.world[self.pos]
end
