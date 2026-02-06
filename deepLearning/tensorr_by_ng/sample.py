import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt 
import tensorflow as tf

# 1. Load and Prepare Data
# Assuming coffee_roasting_data.csv exists with Temperature, Duration, Roast_Quality
data = pd.read_csv("coffee_roasting_data.csv")
x1 = data.Temperature
x2 = data.Duration
y = data.Roast_Quality

features = ['Temperature', 'Duration']
x = data[features]
xn_np = x.to_numpy()

# 2. Normalization
norm = tf.keras.layers.Normalization(axis=-1) 
norm.adapt(xn_np) 
xn = norm(xn_np)

# 3. Data Augmentation (Increasing dataset size for training)
y_np = y.to_numpy().reshape(-1, 1)
xt = np.tile(xn, (500, 1))
yt = np.tile(y_np, (500, 1))

# 4. Reproducibility
random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)

# 5. Build Model
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(3, activation='sigmoid', name='l1'),
    tf.keras.layers.Dense(1, activation='sigmoid', name='l2')
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
)

# 6. Train Model
print("Training model...")
history = model.fit(xt, yt, epochs=20, verbose=0) # Increased epochs slightly for better fit

# 7. Visualization of Predictions
# Create a grid to cover the feature space
t_min, t_max = x1.min() - 5, x1.max() + 5
d_min, d_max = x2.min() - 0.5, x2.max() + 0.5
tt, dd = np.meshgrid(np.linspace(t_min, t_max, 100), 
                     np.linspace(d_min, d_max, 100))

# Normalize the grid points before predicting
grid_points = np.c_[tt.ravel(), dd.ravel()]
grid_points_norm = norm(grid_points)
probs = model.predict(grid_points_norm)
probs = probs.reshape(tt.shape)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the Decision Boundary (The background colors)
# Blue region = Good Roast, Red region = Bad Roast
plt.contourf(tt, dd, probs, levels=[0, 0.5, 1], alpha=0.2, colors=['red', 'blue'])
plt.contour(tt, dd, probs, levels=[0.5], colors='black', linewidths=1)

# Plot the Actual Data Points on top
plt.scatter(x1[y==1], x2[y==1], marker='x', c='blue', label='Actual Good Roast', s=50)
plt.scatter(x1[y==0], x2[y==0], marker='o', c='red', label='Actual Bad Roast', s=50, edgecolors='k', alpha=0.5)

plt.title("Coffee Roasting Decision Boundary\n(Model's Prediction Regions vs Actual Data)")
plt.xlabel("Temperature ($^\circ$C)")
plt.ylabel("Duration (minutes)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Print Weights
print('\nFinal Model Weights:')
print('Layer 1 Weights & Biases:', model.get_layer('l1').get_weights())
print('Layer 2 Weights & Biases:', model.get_layer('l2').get_weights())