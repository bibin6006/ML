import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
#from tensorflow.keras.layers import Dense, Input
#from tensorflow import keras

X_train = np.array([[1.0], [2.0]], dtype=np.float32)           #(size in 1000 square feet)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)       #(price in 1000s of dollars)

model=tf.keras.layers.Dense(units=1,activation='linear') #create a model
per=model.get_weights() #get parameter
#print(per)

#sample trail without traning
#a=model(X_train[0])
a=model(X_train[0].reshape(-1,1))
#print(a)
#print(model.get_weights())


#setting parameter
set_w=np.array([[200]])
set_b=np.array([100])
model.set_weights([set_w,set_b])
#print(model.get_weights())

pre=model(X_train)
pre2=set_w*X_train+set_b
print("model prediction:",pre)
print("equation:",pre2)

sample=np.linspace(1,2,50).reshape(-1,1)
predict=model(sample)
plt.scatter(X_train,Y_train,c='r')
plt.plot(sample,predict)
plt.show()
