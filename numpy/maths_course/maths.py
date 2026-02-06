import numpy as np
arr1=np.array([1,2,3,4,5])
arr2=np.array([1,1,1,1,1])
print(arr1+arr2)
print(arr1-arr2)
print(arr1*arr2)
print(arr1/arr2)

#vector "broadcasting"
#Suppose you need to convert miles to kilometers.
vector=np.array([1,2,3])
print(vector*1.6)