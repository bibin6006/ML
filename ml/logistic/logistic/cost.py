import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))
def cost(x,y,w,b):

    m=x.shape[0]
    cost=0.0
    for i in range(m):
        f_wb_i=np.dot(w,x[i])+b
        f_wb_i=sigmoid(f_wb_i)
        cost+=-y[i]*np.log(f_wb_i)-(1-y[i])*np.log(1-f_wb_i)

    return cost/m    




x = np.array([[0.5, 1.5]
              ,[1,1],
              [1.5, 0.5],
              [3, 0.5], 
              [2, 2],
              [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])
w=np.array([1,1])
b=-3
print(cost(x,y,w,b))