import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler 

x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

scaler=StandardScaler()
x_norm=scaler.fit_transform(x_train)
# print(x_train)
# print(x_norm)

model=SGDRegressor(max_iter=1000)
model.fit(x_norm,y_train)

predict=model.predict(x_norm)
print(y_train)
print(predict)
fig,axis=plt.subplots(nrows=1,ncols=4)
print(len(axis))
title=['size','bedrooms','floors','age']
for i in range(len(axis)):
    axis[i].scatter(x_train[:,i],y_train)
    axis[i].scatter(x_train[:,i],predict)
    axis[i].set_xlabel(title[i])
    axis[i].set_ylabel('price')

plt.show()