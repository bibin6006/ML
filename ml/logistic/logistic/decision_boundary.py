import numpy as np
import matplotlib.pyplot as plt


x = np.array([[0.5, 1.5]
              ,[1,1],
              [1.5, 0.5],
              [3, 0.5], 
              [2, 2],
              [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

# in this b=-3 and w1=1,w2=1
#x1+x2-3=0
#x1=3-x2

pos=y==1
neg=y==0
# print(pos)
# print(neg)
x0=x[:,1]
x1=3-x0
print(x1)

plt.plot(x0,x1)
plt.scatter(x[pos,0],x[pos,1],marker='x',s=300,c='r')
plt.scatter(x[neg,0],x[neg,1],marker='o',s=300,c='b')
plt.show()

