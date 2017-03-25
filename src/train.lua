require "nn"
require 'csvigo'
require "io"
require "optim"
require "torch"

-- Read data from CSV to tensor
local csvFile = io.open('../data/processed/train_data.csv', 'r')  
local s = 0
for line in csvFile:lines('*l') do  
  s = s+1
end
csvFile:close() 

local csvFile = io.open('../data/processed/train_data.csv', 'r')  
local header = csvFile:read()

local i = 0  
train_data = torch.Tensor(s, 33)
for line in csvFile:lines('*l') do  
  print(line)
  i = i + 1
  local l = line:split(',')
  for key, val in ipairs(l) do
     train_data[i][key] = val
  end
end
csvFile:close() 
dataset_inputs = train_data[{ {},{1,32} }]
dataset_outputs  = train_data[{ {},{33,33} }]



local csvFile = io.open('../data/processed/test_data.csv', 'r')  
local s = 0
for line in csvFile:lines('*l') do  
  s = s+1
end
csvFile:close() 
test_data = torch.Tensor(s, 33)

local csvFile = io.open('../data/processed/test_data.csv', 'r')  
local header = csvFile:read()

local i = 0  
for line in csvFile:lines('*l') do  
  i = i + 1
  local l = line:split(',')
  for key, val in ipairs(l) do
    test_data[i][key] = val
  end
end
csvFile:close() 
X_test = test_data[{ {},{1,32} }]
Y_test = test_data[{ {},{33,33} }]


local csvFile = io.open('../data/processed/to_submit.csv', 'r')  

local s = 0
for line in csvFile:lines('*l') do  
  s = s+1
end
csvFile:close() 
to_submit = torch.Tensor(s, 32)


local csvFile = io.open('../data/processed/to_submit.csv', 'r')
local header = csvFile:read()


local i = 0  
for line in csvFile:lines('*l') do  
  i = i + 1
  local l = line:split(',')
  for key, val in ipairs(l) do
    to_submit[i][key] = val
  end
end

csvFile:close() 


--Neural network design.
model = nn.Sequential()
model:add(nn.Linear(32,5))
model:add(nn.ReLU())
model:add(nn.Linear(5,1))
model:add(nn.Sigmoid())

criterion = nn.BCECriterion()



--Update function

x, dl_dx = model:getParameters()
feval = function(x_new)
   if x ~= x_new then
      x:copy(x_new)
   end

   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > dataset_inputs:size(1) then _nidx_ = 1 end

   local inputs = dataset_inputs[_nidx_]
   local target = dataset_outputs[_nidx_]

   dl_dx:zero()

   local outputs = model:forward(inputs)
   local loss_x = criterion:forward(outputs, target)
   local dl_do = criterion:backward(outputs, target)
   model:backward(inputs, dl_do)
    
   return loss_x, dl_dx
end

--Set parameters
sgd_params = {
   learningRate = 1e-3,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}
epochs = 500


--Train model
for i = 1,epochs do
   current_loss = 0
   for i = 1,dataset_inputs:size(1) do
      _,fs = optim.sgd(feval,x,sgd_params)
      current_loss = current_loss + fs[1]
   end

   current_loss = current_loss / dataset_inputs:size(1)
   print('epoch = ' .. i .. 
	 ' of ' .. epochs .. 
	 ' current loss = ' .. current_loss)
end
print(current_loss)


--Test
local output =  model:forward(X_test)
local restable = torch.totable(output)
csvigo.save("../data/result/result.csv", restable)
-- local lb = torch.gt(output, 0.5)
-- lb = lb:double() 
-- local acc = torch.eq(lb, Y_test)
-- local accuracy = acc:sum()/X_test:size(1)


--Result to submit

local output =  model:forward(to_submit)
local restable1 = torch.totable(output)
csvigo.save("../data/result/to_submit_result.csv", restable1)
