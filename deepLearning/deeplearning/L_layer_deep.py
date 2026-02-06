import numpy as np
import copy 
import matplotlib.pyplot as plt


X=np.array([[0.5,1,1.5,3,2,1],
           [1.5,1,0.5,0.5,2,2.5]])
Y= np.array([[0, 0, 0, 1, 1, 1]]) 

def initialize_parameters(layer_dim):
    L=len(layer_dim)
    parameters={}
    for l in range(1,L):
        parameters['W'+str(l)]=np.random.randn(layer_dim[l],layer_dim[l-1])*0.01
        parameters['b'+str(l)]=np.zeros((layer_dim[l],1))
    return parameters 

def cost(AL,Y):
    m=Y.shape[1]
    logprobs = np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), (1 - Y))
    c = -np.sum(logprobs) / m
    return c
def linear_function(A,W,b):
    Z=np.dot(W,A)+b
    cache=(A,W,b)
    return Z,cache      

def sigmoid(Z):
    s=1/(1+np.exp(-Z))
    return s,Z

def relu(Z):
    r=np.maximum(0,Z)
    return r,Z

def activaton_function(A,W,b,activation):
    if activation=='relu':
        Z,linear_cache=linear_function(A,W,b)
        AL,active_cache=relu(Z)
    if activation=='sigmoid':
        Z,linear_cache=linear_function(A,W,b)
        AL,active_cache=sigmoid(Z)

    cache=(linear_cache,active_cache)
    return AL,cache

def forward_propagation(X,Y,parameters):
    A=X
    caches=[]
    L=len(parameters)//2
    for l in range(1,L):
        pre_a=A
        A,cache=activaton_function(pre_a,parameters['W'+str(l)],parameters['b'+str(l)],'relu')
        caches.append(cache)
    AL,cache=activaton_function(A,parameters['W'+str(L)],parameters['b'+str(L)],'sigmoid')
    c=cost(AL,Y)
    caches.append(cache)
    return AL,caches,c

def linear_backward(dZ,cache):
    A_pre,W,b=cache
    m=A_pre.shape[1]
    dW=np.dot(dZ,A_pre.T)/m
    db=np.sum(dZ,axis=1,keepdims=True)/m
    dA=np.dot(W.T,dZ)
    return dA,dW,db

def relu_derivative(dA,x):
    """
    Computes the derivative of ReLU for a single value or a NumPy array.
    Returns 1 for values > 0 and 0 otherwise.
    """
    dg=(x > 0).astype(float)
    return dA*dg

def sigmoid_derivative(dA,x):
    """
    Compute the derivative of the sigmoid function.
    Note: 'x' here is the original input value.
    """
    s,_= sigmoid(x)
    dg=s * (1 - s)
    return dA*dg
def linear_activation_backward(dA,cache,activation):

    linear_cache,activation_cache=cache
    if activation=='sigmoid':
        dZ=sigmoid_derivative(dA,activation_cache)
        dA_pre,dW,db=linear_backward(dZ,linear_cache)

    if activation=='relu':
        dZ=relu_derivative(dA,activation_cache)
        dA_pre,dW,db=linear_backward(dZ,linear_cache)


    return  dA_pre,dW,db

def L_model_backward(AL,Y,caches):
    # grads={}
    # L=len(caches)
    # dAL=-(np.divide(Y,AL)-np.divide((1-Y),(1-AL)))
    # current_cache=caches[-1]
    # dA_pre,dW,db=linear_activation_backward(dAL,current_cache,'sigmoid')
    # grads['dA'+str(L-1)]=dA_pre
    # grads['dW'+str(L)]=dW
    # grads['db'+str(L)]=db

    # for l in range(L-1,0,-1):
    #     current_cache=caches[l-1]
    #     dA_pre,dW,db=linear_activation_backward(grads['dA'+str(l)],current_cache,'relu')
    #     grads['dA'+str(l-1)]=dA_pre
    #     grads['dW'+str(l)]=dW
    #     grads['db'+str(l)]=db
    # return grads    
    grads = {}
    L = len(caches) # Number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # Ensure Y is same shape as AL
    
    # 1. Initialize backprop: Gradient of cost with respect to AL
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # 2. Output layer (SIGMOID -> LINEAR) gradients
    current_cache = caches[L-1]
    # We store the dA for the previous layer (L-1)
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = \
        linear_activation_backward(dAL, current_cache, "sigmoid")
    
    # 3. Hidden layers (RELU -> LINEAR) gradients
    for l in reversed(range(L-1)):
        # l goes from L-2 down to 0
        current_cache = caches[l]
        
        # We take dA from the layer we just processed (l+1) 
        # to calculate gradients for the current layer (l+1)
        dA_prev_temp, dW_temp, db_temp = \
            linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads
        

def update_parameters(params, grads, learning_rate):
    
    parameters = copy.deepcopy(params)

    L = len(parameters) // 2

    for l in range(L):
        parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
        parameters["b"+str(l+1)]=parameters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)]
        
    return parameters
     



def model(layers_dim,X,Y,learning_rate=0.5,iteration=1000):
    params=initialize_parameters(layers_dim)
    cost=[]

    for i in range(iteration):
        AL,caches,c=forward_propagation(X,Y,params)
        cost.append(c)
        grads=L_model_backward(AL,Y,caches)
        params=update_parameters(params, grads, learning_rate)

        if i%100==0:
            print(f"cost:{c}")
    return cost  







layers_dim=[2,4,1]
c=model(layers_dim,X,Y)

s=np.arange(1,1001,1)
plt.plot(s,c)
plt.show()

