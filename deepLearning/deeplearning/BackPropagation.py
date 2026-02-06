import numpy as np


X=np.array([[0.5,1,1.5,3,2,1],
           [1.5,1,0.5,0.5,2,2.5]])
y = np.array([[0, 0, 0, 1, 1, 1]])
print(X.shape)
print(y.shape)

def initialize_para(dim):
    w=np.zeros((dim,1))
    b=0.0
    return w,b

def sigmoid(X):
    return 1/(1+np.exp(-X))

def propagate(w,b,X,y):
    m=X.shape[1]
    #sigmoid function w size(n,1) w transpose size(1,n) x size(n,m) so A size (1,m)
    #n is the number of features and m is the number of example 
    A=sigmoid(np.dot(w.T,X)+b)

    #cost function 
    cost=-1*np.sum(y*np.log(A)+(1-y)*np.log(1-A))/m
    
    #derivative of w and b using backpropagation 

    #x size (n,m) and A-y size (1,m) so A-y transpose size (m,1) so dw size (n,1)
    dw=np.dot(X,(A-y).T)/m
    db=np.sum(A-y)/m

    grads={'dw':dw,'db':db}
    return grads,cost 

def gradient_descent(w,b,X,y,iteration,learning_rate):
    cost=[]
    for i in range(iteration):
      
        para,c=propagate(w,b,X,y)
        dw=para['dw']
        db=para['db']
        cost.append(c)
        w=w-learning_rate*dw
        b=b-learning_rate*db

        if i%100==0:
            print(f"cost:{cost[-1]} weghts:{w} bais:{b}")
            print()
    para={'w':w,'b':b}
    grads={'dw':dw,'db':db}   
    return para,grads

def predicte(w,b,x):

    A=sigmoid(np.dot(w.T,x)+b)
    pre=np.zeros((1,x.shape[1]))
    for i in range(x.shape[1]):
        if A[0,i]>0.5:
            pre[0,i]=1
        else:
            pre[0,i]=0

    return pre 


w,b=initialize_para(X.shape[0])
para,grads=gradient_descent(w,b,X,y,1000,0.1)
pre=predicte(para['w'],para['b'],X)
print(f"actual value:{y}")
print(f"predicted value:{pre}")
    

