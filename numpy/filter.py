#find the even numbers from an array using filter technique

import numpy as np

arr=np.array([1,2,3,4,5,6,7,8,9,10])
'''
filter_arr=[]

for i in arr:
    if i%2==0:
        filter_arr.append(True)
    else:
        filter_arr.append(False)  

res=arr[filter_arr] 
print(res)
'''
# or 


filter_arr=arr%2==0
print(arr[filter_arr])
