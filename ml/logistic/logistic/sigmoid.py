import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
#exp() is used to calculste exponential (e^z)
# print(np.exp(1))
# print(np.exp(np.array([1,2,3,4])))

def sigmoid(z):
    return 1/(1+np.exp(-z))

z=np.arange(-10,11)
print(z)
y=sigmoid(z)

print(f"{z}  {y}")

plt.plot(z,y)
plt.show()

