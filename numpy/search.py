#where() method help to find the index of certain value

import numpy as np
arr=np.array([1,2,3,4,4,5])
res=np.where(arr==4)
print(res)

# find the even numbers from an array using where() method

idx=np.where(arr%2==0)
for i in idx:
    print(arr[i])

    