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
print(f"Temperature after normalization: max: {np.max(xn[:, 0]):.2f} and min: {np.min(xn[:, 0]):.2f}")
print(f"Duration after normalization: max: {np.max(xn[:, 1]):.2f} and min: {np.min(xn[:, 1]):.2f}")

#creating a copy 
y_np=y.to_numpy().reshape(-1,1)
xt=np.tile(xn,(1000,1))#10 is row 1 is column 
yt=np.tile(y_np,(1000,1))
print(yt.shape)
#to remove randomness get same output every time 
# random.seed(1234)
# np.random.seed(1234)
tf.random.set_seed(1234)
# create a model with two layers
model=tf.keras.models.Sequential([
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(3,activation='sigmoid',name='l1'),
        tf.keras.layers.Dense(1,activation='sigmoid',name='l2')

])
w1,b1=model.get_layer('l1').get_weights()
print(f"w:{w1}  b:{b1}")

model.compile(
    
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1),
)

model.fit(
    xt,yt,            
    epochs=5,
)
sample_x1=np.linspace(1.71,-1.93,40)
sample_x2=np.linspace(1.70,-1.71,40)
sample_x=np.column_stack((sample_x1, sample_x2))
predict=np.zeros(40)
pre=model.predict(sample_x)
for i in range(len(pre)):
    if pre[i]>=0.5:
        predict[i]=1
    else:
        predict[i]=0    

print(f"actual value:{yt}")
print(f"predicted value:{predict}")
print('layer1')
print(model.get_layer('l1').get_weights())
print('layer2')
print(model.get_layer('l2').get_weights())



neg1=predict==0
pos1=predict==1
plt.scatter(x1[pos1],x2[pos1],marker='x',c='b',s=200)
plt.scatter(x1[neg1],x2[neg1],marker='o',c='r',s=200)
# #plt.scatter(x1[pos],x2[pos],marker='x',c='b',s=100)#good coffee
# #plt.scatter(x1[neg],x2[neg],marker='o',c='r',s=100)
plt.show()
