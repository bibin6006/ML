import numpy as np
# arr=np.arange(0,10,1)
# print(arr)
# What if you wanted to create an array with five evenly spaced values in the interval from 0 to 100?
arr=np.linspace(0,100,5,dtype=int)# fist parameter is start point and second parameter is ending point s exclude and third parameer is number of element in array
print(arr)