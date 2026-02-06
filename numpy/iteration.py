#nditer() is a function that help to iterate an array

import numpy as np
arr=np.array([[1,2],[3,4]])
for i in np.nditer(arr):
    print(i)

#ndenumerate() function while iterating returns its index and value together
for key,val in np.ndenumerate(np.nditer(arr)):
    print(key,val)
