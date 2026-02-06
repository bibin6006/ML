import numpy as np
def myadd(x,y):
    return x+y

myadd=np.frompyfunc(myadd,2,1)#first parameter is function name ,second number of function argument,third number of output
x=[1,2,3,4]
y=[10,10,10,10]

print(myadd(x,y))