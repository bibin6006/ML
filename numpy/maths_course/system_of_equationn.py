import numpy as np
import matplotlib.pyplot as plt
plt.plot_lines()
#−𝑥1+3𝑥2=7,3𝑥1+2𝑥2=1
#  -1 3  x1  =7
#   3 2  x2  =1 

A=np.array([[-1,3],[3,2]])
b=np.array([7,1])
X=np.linalg.solve(A,b)# it return an array with two element first vslue is x1 and second value is x2
print(X) 

# to find determinant
d=np.linalg.det(A)
print(d)# if the value is non zero then it is non singular menas it have unique solution /if the value is zero than it is singular means it have no unique solution or infinitly many solution 