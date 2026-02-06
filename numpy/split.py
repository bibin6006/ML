# array_split() function use to split an array
import numpy as np
arr=np.array([1,2,3,4,5,6])
res=np.array_split(arr,3)
#print(res)
#print(res[0])

arr1=np.array([[1,2],[3,4]])
res1=np.array_split(arr1,2)
res2=np.array_split(arr1,2,axis=1)
print(res1)
print(res2)
