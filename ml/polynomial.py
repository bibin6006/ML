# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import SGDRegressor

# x=np.arange(1,3,1)
# y=1+x**2
# x=x.reshape(-1,1)
# print(x)
# print(y)

# model=SGDRegressor(max_iter=1000)
# # model.fit(x,y)
# # sample=np.linspace(1,19)
# # sample=sample.reshape(-1,1)
# # pre=model.predict(sample)

# # plt.scatter(x,y,marker='x',c='r')
# # plt.plot(sample,pre)
# # plt.show()

# x=x.reshape(-1,1)
# new_x=x**2
# print(new_x)
# model.fit(new_x,y)
# pre=model.predict(new_x)

# plt.scatter(x,y,marker='x',c='r')
# plt.plot(x,pre)
# plt.show()


 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler  # Import this!

x = np.arange(1, 20, 1)
y = 1 + x**2
x = x.reshape(-1, 1)

# Create the feature x^2
new_x = x**2
new_x = new_x.reshape(-1, 1)

# --- STEP 1: Scale the data ---
scaler = StandardScaler()
new_x_norm = scaler.fit_transform(new_x)  # Compute mean/std and transform

# --- STEP 2: Train on Scaled Data ---
model = SGDRegressor(max_iter=1000)
model.fit(new_x_norm, y)

# --- STEP 3: Predict using Scaled Data ---
pre = model.predict(new_x_norm)

# --- Plotting ---
plt.scatter(x, y, marker='x', c='r', label="Actual")
plt.plot(x, pre, label="Predicted") # Plot original x vs predictions
plt.legend()
plt.show()
