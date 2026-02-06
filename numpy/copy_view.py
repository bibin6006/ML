#copy owns the data and view don't owns the data
# any change made in the original array didn't affect copy and any change made in a copy didn't affect original array
# any change made in the original array is affected by view
import numpy as np
arr=np.array([1,2,3,4])
x=arr.copy()
y=arr.view()
arr[0]=99
print(x)
print(y)

#base is use to check an array owns a data
#if it owns base returns None otherwise returns original arrary
print(x.base)
print(y.base)