--gfx = require 'gfx.js'
m = require 'manifold'


local function show_scatter_plot(mapped_x, labels, K, positions)
  print(K)

  -- count label sizes:
  local cnts = torch.zeros(K)
  for n = 1,labels:nElement() do
    cnts[labels[n]] = cnts[labels[n]] + 1
  end
  
  print(cnts)

  -- separate mapped data per label:
  local mapped_data = {}
  for k = 1,K do
    mapped_data[k] = { key = positions[k], values = torch.Tensor(cnts[k], 2) }
  end
  local offset = torch.Tensor(K):fill(1)
  for n = 1,labels:nElement() do
    mapped_data[labels[n]].values[offset[labels[n]]]:copy(mapped_x[n])
    offset[labels[n]] = offset[labels[n]] + 1
  end

  -- show results in scatter plot:
  local gfx = require 'gfx.js'
  gfx.chart(mapped_data, {
     chart = 'scatter',
     width = 800,
     height = 800,
  })
end

local function calc_embedding(fin, fout)
  local f = torch.DiskFile(fin, 'r')
  local o = f:readObject()
  f:close()

  local opts = {ndims = 2, perplexity = 30 }
  o.x = m.embedding.tsne(o.x, opts)
  
  local f = torch.DiskFile(fout, 'w')
  f:writeObject(o)
  f:close()
end

local function show_mapping(fn)
  local f = torch.DiskFile(fn, 'r')
  local o = f:readObject()
  f:close()
  print(o.l)
  show_scatter_plot(o.x, o.l, o.k, o.p)
end

torch.mm(torch.rand(100,100), torch.rand(100,100))
print('test')

calc_embedding('sampling.t7', 'mapping3.t7')
show_mapping('mapping3.t7')
