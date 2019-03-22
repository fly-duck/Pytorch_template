import torch as t 



#autograd 

# x=t.ones(2,2,requires_grad=True)
x=t.tensor([[1.,-1.],[1.,1.]],requires_grad=True)
print(x)

# y=2*x.sum()
# bad:grad can be implicitly created only for scalar outputs (or it will be a jaccobian)
# y=x.pow(2) y=[[x1^2,x2^2],[x3^2,x4^2]] 
# y=x.pow(2).sum()   

y=x.pow(5).sum()   

print(y)


y.grad_fn

y.backward(retain_graph=True)

print(x.grad)

# keep graph accumulating 
# y=x.sum()  y=[[x1,x2],[x3,x4]]

y.backward()
print(x.grad)
