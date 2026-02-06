#we can use dtype with array() function it is a optional argument 
import numpy as np
arr=np.array([1,2,3,4,5],dtype="S")
print(arr)
print(arr.dtype)


#astype is method that convert data type of an existing array 
arr=np.array([1,2,3,4,])
a=arr.astype('f')
print(a)
print(a.dtype)