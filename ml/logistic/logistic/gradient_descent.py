import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

def sigmoid(z):
    return 1/(1+np.exp(-z))
    
def cost(x,y,w,b):
    m=x.shape[0]
    cost=0.0
    for i in range(m):
        z=np.dot(x[i],w)+b
        f_wbi=sigmoid(z)

        cost+=-y[i]*np.log(f_wbi)-(1-y[i])*np.log(1-f_wbi)
    return cost/m 

def compute_gradient(x,y,w,b):
    m,n=x.shape
    dj_dwi=np.zeros(n)
    dj_dbi=0.0   
    for i in range(m):
        z=np.dot(x[i],w)+b
        f_wbi=sigmoid(z)
        err=f_wbi-y[i]
        for j in range(n):
            dj_dwi[j]+=err*x[i,j]
        dj_dbi+=err

    dj_dw=dj_dwi/m
    dj_db=dj_dbi/m
    return dj_dw,dj_db

def gradient(x,y,w,b,alpha,iteration):
    history=[]
    for i in range(iteration):

        dj_dw,dj_db=compute_gradient(x,y,w,b)

        w=w-(alpha*dj_dw)
        b=b-(alpha*dj_db) 

        history.append(cost(x,y,w,b))

        if i% 1000==0:
            print(f"iteration:{i:4} and cost:{history[-1]}")
    
    return w,b

iteration=10000
alpha=0.1
n=X_train.shape[1]
w=np.zeros(n)
b=0.0
final_w,final_b=gradient(X_train,y_train,w,b,alpha,iteration)
print(f"final w:{final_w} and final_b:{final_b}")
# print(X_train.shape[1])
# print(w.shape[0])

x0=-final_b/final_w[0]
x1=-final_b/final_w[1]


pos=y_train==1
neg=y_train==0
# print(pos)
# print(neg)


plt.plot([x0,0],[0,x1])
plt.scatter(X_train[pos,0],X_train[pos,1],marker='x',s=300,c='r')
plt.scatter(X_train[neg,0],X_train[neg,1],marker='o',s=300,c='b')
plt.show()

