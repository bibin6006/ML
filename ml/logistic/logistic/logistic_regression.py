import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression

x_train = np.array([0., 1, 2, 3, 4, 5]).reshape(-1,1)
y_train = np.array([0,  0, 0, 1, 1, 1])
model=LogisticRegression()
model.fit(x_train,y_train)

sample=np.linspace(0,6,100).reshape(-1,1)
pre=model.predict(sample)

pos=y_train==1
neg=y_train==0


plt.scatter(x_train[neg],y_train[neg],marker='o',s=300,c='b')
plt.scatter(x_train[pos],y_train[pos],marker='x',s=300,c='r')
plt.plot(sample,pre)
plt.show()