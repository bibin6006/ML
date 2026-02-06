
from numpy import random
#randint() method create random integer from 0 to user choice

x=random.randint(100,size=(5,3))
print(x)

#rand() method create random floating point numbers from 0 to 1
y=random.rand(5)
print(y)

#choice() method create random numbers from given array
z=random.choice([1,2,3,4,5],size=(3,4))
print(z)