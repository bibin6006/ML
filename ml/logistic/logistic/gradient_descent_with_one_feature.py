import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost(x,y,w,b):
    m=x.shape[0]
    cost=0.0
    for i in range(m):
        z=x[i]*w+b
        f_wbi=sigmoid(z)
        cost+=-y[i]*np.log(f_wbi)-(1-y[i])*np.log(1-f_wbi)

    return cost/m

def compute_gradient(x,y,w,b):
    dj_dwi=0.0
    dj_dbi=0.0
    m=x.shape[0]
    for i in range(m):
        z=x[i]*w+b
        f_wb=sigmoid(z)
        dj_dwi+=(f_wb-y[i])*x[i]
        dj_dbi+=f_wb-y[i]

    dj_dw=dj_dwi/m
    dj_db=dj_dbi/m
    return dj_dw,dj_db

def gradient(x,y,w,b,alpha,iteration):
    history=[]
    for i in range(iteration):

        dj_dw,dj_db=compute_gradient(x,y,w,b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db

        history.append(cost(x,y,w,b))

        if i%1000==0:
            print(f"iteration:{i:4} and cost:{history[-1]}")

    return w,b        


x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

alpha=1
iteration=10000
w=0.0
b=0.0
final_w,final_b=gradient(x_train,y_train,w,b,alpha,iteration)
print(f"final w:{final_w} and final b:{final_b}")

def predict(x,w,b):
    m=x.shape[0]
    y=np.zeros(m)
    for i in range(m):
        z=x[i]*w+b
        y[i]=sigmoid(z)

    return y    

pos=y_train==1
neg=y_train==0
# print(pos)
# print(neg)

sample=np.linspace(0,6,100)

predicted_y=predict(sample,final_w,final_b)

plt.scatter(x_train[neg],y_train[neg],marker='o',s=300,c='b')
plt.scatter(x_train[pos],y_train[pos],marker='x',s=300,c='r')
plt.plot(sample,predicted_y)
plt.show()