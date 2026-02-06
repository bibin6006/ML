import numpy as np
import math as m

x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

#calculate cost function
def cost(x,y,w,b):
    m=x.shape[0]
    cost=0
    for i in range(m):
        f_wb=(np.dot(w,x[i]))+b
        cost+=(f_wb-y[i])**2
    total_cost=(1/(2*m))*cost
    return total_cost

#calculate derivative of gradient descent

def compute_gradient(x,y,w,b):
    m,n=x.shape
    dj_dw=np.zeros(n)
    dj_db=0
    for i in range(m):
        err=(np.dot(w,x[i])+b)-y[i]
        for j in range(n):
            dj_dw[j]+=err*x[i,j]
        dj_db+=err
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db    
    
#calculate gradient descent
def gradeint_descent(x,y,w,b,alpha,iteration):
    j=[]
    for i in range(iteration):
        dj_dw,dj_db=compute_gradient(x,y,w,b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db
        if i<iteration:
            j.append(cost(x,y,w,b))
        if i%m.ceil(iteration/10)==0:
            print(f"iteraion:{i:4} cost:{j[-1]} w:{w} b:{b}")

    return w,b        


def z_score_normalize(x):
    mu=np.mean(x,axis=0)
    sigma=np.std(x,axis=0)
    x_norm=(x-mu)/sigma
    return x_norm


def predicte(x,y,fnal_w,final_b):
    m=x.shape[0]
    for i in range(m):
        print(f"prediction:{np.dot(final_w,x[i])+final_b} actual value:{y[i]}")



init_w=np.zeros(4,dtype='float')
init_b=0.0
alpha=0.01
iteration=1000
print("without scaling:\n")
final_w,final_b=gradeint_descent(x_train,y_train,init_w,init_b,alpha,iteration)
predicte(x_train,y_train,final_w,final_b)

print("with sacling:\n")

x_norm=z_score_normalize(x_train)
final_w,final_b=gradeint_descent(x_norm,y_train,init_w,init_b,alpha,iteration)
predicte(x_norm,y_train,final_w,final_b)





