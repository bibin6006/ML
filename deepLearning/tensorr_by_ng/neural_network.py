import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt 
import tensorflow as tf

data=pd.read_csv("coffee_roasting_data.csv")
x1=data.Temperature
x2=data.Duration
y=data.Roast_Quality
pos=y==1
neg=y==0

# plt.scatter(x1[pos],x2[pos],marker='x',c='b',s=100)#good coffee
# plt.scatter(x1[neg],x2[neg],marker='o',c='r',s=100)
# plt.show()
features=['Temperature','Duration']
x=data[features]
print(x.shape)
print(y.shape)

print(f"temperature before normalization:max:{x['Temperature'].max()} and min:{x['Temperature'].min()}")
print(f"Duration before normalization;max:{x['Duration'].max()} and min: {x['Duration'].min()}")
xn_np = x.to_numpy()
norm=tf.keras.layers.Normalization(axis=-1) 
norm.adapt(xn_np) #learn mean and variance
xn=norm(xn_np)
# print(f"Temperature after normalization: max: {np.max(xn[:, 0]):.2f} and min: {np.min(xn[:, 0]):.2f}")
# print(f"Duration after normalization: max: {np.max(xn[:, 1]):.2f} and min: {np.min(xn[:, 1]):.2f}")

#creating a copy 
y_np=y.to_numpy().reshape(-1,1)
xt=np.tile(xn,(5000,1))#10 is row 1 is column 
yt=np.tile(y_np,(5000,1))
print(xt.shape)
print(yt.shape)


#sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

def Dense(a_in,W,b):
    units=W.shape[1]
    a_out=np.zeros(units)
    for j in range(units):
        w=W[:,j]
        z=np.dot(a_in,w)+b[j]
        a_out[j]=sigmoid(z)
    return a_out    

def Sequential(x,W1,W2,b1,b2):
    a1=Dense(x,W1,b1)
    a2=Dense(a1,W2,b2)

    return a2

W1=np.array([[ 13.66818 , 106.75266 ,  40.26214 ],
       [-26.775103,   6.292546,  36.20732 ]])
W2=np.array([[  5.3219123],
       [-39.56042  ],
       [-23.539095 ]])

b1=np.array([ 4.4367752e+01, -2.4814655e+01, -2.9216392e-03])
b2=np.array([-5.0809755])

def predict(X,W1,W2,b1,b2):
    m=X.shape[0]
    pre=np.zeros((m,1))
    for i in range(m):
        pre[i]=Sequential(X[i],W1,W2,b1,b2)
    return pre

print(predict(xt[:3],W1,W2,b1,b2))
