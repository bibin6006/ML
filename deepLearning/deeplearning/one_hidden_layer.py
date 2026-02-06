import numpy as np

X=np.array([[0.5,1,1.5,3,2,1],
           [1.5,1,0.5,0.5,2,2.5]])
y = np.array([[0, 0, 0, 1, 1, 1]])    

def sigmoid(z):
    return 1/(1+np.exp(-z))

def layer_size(X,y):
    #number of features
    n_x=X.shape[0]
    
    #number of hidden layer
    n_h=4

    #number of output layer
    n_y=y.shape[0]

    return n_x,n_h,n_y
def initialize_paramter(n_x,n_h,n_y):
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))

    para={'W1':W1,'b1':b1,'W2':W2,'b2':b2}
    return para 
def cost(X,y,cache):
    m=X.shape[1]
    A2=cache['A2']
    logprobs = np.multiply(np.log(A2), y) + np.multiply(np.log(1 - A2), (1 - y))
    c = -np.sum(logprobs) / m
    return c

def forword_propagation(X,parameters):
    W1=parameters['W1']
    b1=parameters['b1']
    W2=parameters['W2']
    b2=parameters['b2']


    Z1=np.dot(W1,X)+b1 #w1(4,2) X(2,m) Z1(4,m) b1(4,1)
    A1=np.tanh(Z1) #A1(4,m)
    Z2=np.dot(W2,A1)+b2 #w2(1,4) A1(4,m) b2(1,1) z2(1,m)
    A2=sigmoid(Z2) #A2(1,m)

    cache={'Z1':Z1,'A1':A1,'Z2':Z2,'A2':A2}
    c=cost(X,y,cache)
    return cache,c
def backword_propagation(X,y,parameters):
    m=X.shape[1]
    cache,cost=forword_propagation(X,parameters)
    W2=parameters['W2']

    Z1=cache['Z1']
    A1=cache['A1']
    Z2=cache['Z2']
    A2=cache['A2']

    dz2=A2-y #A2(1,m) y(1,m) dz2(1,m)
    dw2=np.dot(dz2,A1.T)/m #dz2(1,m) A(1,m) A.T(m,1) dw2(1,1)
    db2=np.sum(dz2,axis=1,keepdims=True)/m
    dz1=np.dot(W2.T,dz2)*(1-A1**2) #w2(1,4) w2.T(4,1) dz2(1,m)  A1(4,m) dz1(4,m)
    dw1=np.dot(dz1,X.T)/m #dz1(4,m) x(2,m) x.T(m,2) dw1(4,2)
    db1=np.sum(dz1,axis=1,keepdims=True)/m
    grads={'dw2':dw2,'db2':db2,'dw1':dw1,'db1':db1,}
    return grads,cost




def update_parameters(W1,b1,W2,b2,X,y,learning_rate,iteration):
    cost=[]
    parameters={'W1':W1,'b1':b1,'W2':W2,'b2':b2}
    for i in range(iteration):
        
        grads,c=backword_propagation(X,y,parameters)
        cost.append(c)
        dw1=grads['dw1']
        db1=grads['db1']
        dw2=grads['dw2']
        db2=grads['db2']
        W1=W1- learning_rate*dw1
        b1=b1-learning_rate*db1
        W2=W2-learning_rate*dw2
        b2=b2-learning_rate*db2


        parameters['W1']=W1
        parameters['b1']=b1
        parameters['W2']=W2
        parameters['b2']=b2

        if i%100==0:
            print(f"cost:{cost[-1]}")
    return parameters


def predict(X,parameters):
    cache,c=forword_propagation(X,parameters)
    A2=cache['A2']
    pre=(A2 > 0.5).astype(int)
    return pre
n_x,n_h,n_y=layer_size(X,y)
para=initialize_paramter(n_x,n_h,n_y)            
parameters=update_parameters(para['W1'],para['b1'],para['W2'],para['b2'],X,y,0.1,1000)  
pre=predict(X,parameters) 
print(f"actual value :{y}")
print(f"predicted vlue : {pre}")


