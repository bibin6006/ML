
# we can join two or more array using concatenate() function

import numpy as np
arr1=np.array([[1,2],[3,4]])
arr2=np.array([[5,6],[7,8]])
new=np.concatenate((arr1,arr2),axis=1)
#print(new)
new1=np.hstack((arr1,arr2))
print(new1)
new2=np.vstack((arr1,arr2))
print(new2)

arr3=np.array([1,2,3,4])
arr4=np.array([5,6,7,8])

new3=np.hstack((arr3,arr4))
print(new3)
new4=np.vstack((arr3,arr4))
print(new4)
