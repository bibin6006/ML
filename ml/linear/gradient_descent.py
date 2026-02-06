import numpy as np
import math as m

#calculate cost function
def cost(x,y,w,b):
    m=x.shape[0]
    cost=0
    for i in range(m):
        f_wb=w*x[i]+b
        cost+=(f_wb-y[i])**2
    total_cost=1/(2*m)*cost
    return total_cost


#calculate the derivative 
def compute_gradient(x,y,w,b):
    m=x.shape[0]
    dj_db_i=0
    dj_dw_i=0
    for i in range(m):
        f_wb=w*x[i]+b
        dj_dw_i+=(f_wb-y[i])*x[i]
        dj_db_i+=(f_wb-y[i])
    dj_dw=dj_dw_i/m
    dj_db=dj_db_i/m
    return dj_dw,dj_db
#calculate gradient descent
def gradient_descent(x,y,w,b,alpha,iteration):
    j_history=[]
    p_history=[]
    for i in range(iteration):

        dj_dw,dj_db=compute_gradient(x,y,w,b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db
        if i<iteration:
            j_history.append(cost(x,y,w,b))
            p_history.append([w,b])
        if i%m.ceil(iteration/10)==0:
            print(f"iteration:{i:4} cost:{j_history[-1]}  w:{p_history[-1][0]}  b:{p_history[-1][1]}")
    return p_history[-1]        


        
x_train=np.array([1000,2000])
y_train=np.array([300000,500000])
w_init=0
b_init=0
iteration=100000
alpha=1.0e-2 #0.01

#feature scaling 

#z-score normalization 
def z_score_normalize(x_train):
    mu=np.mean(x_train)
    sigma=np.std(x_train)
    x_norm=(x_train-mu)/sigma

    return x_norm,mu,sigma

x_norm,mu,sigma=z_score_normalize(x_train)
final_p=gradient_descent(x_norm,y_train,w_init,b_init,alpha,iteration)


#predicte

def predicte(x,y,final_p):
    m=x.shape[0]
    w=final_p[0]
    b=final_p[1]
    for i in range(m):
        f_wb=w*x[i]+b
        print(f"actual value:{y[i]} predicted value:{f_wb}")

predicte(x_norm,y_train,final_p)        
