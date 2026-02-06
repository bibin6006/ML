import numpy as np
import tensorflow as tf 
from sklearn.datasets import make_blobs


# make  dataset for example
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=1000, centers=centers, cluster_std=1.0,random_state=30)
print(X_train[:5])
print(y_train.shape)

# model=tf.keras.models.Sequential([
#     tf.keras.layers.Dense(25,activation='relu',name='l1'),
#     tf.keras.layers.Dense(15,activation='relu',name='l2'),
#     tf.keras.layers.Dense(4,activation='softmax',name='l3')
# ])

# model.compile(
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     optimizer=tf.keras.optimizers.Adam(0.001),
# )

# model.fit(
#     X_train,y_train,
#     epochs=10
# )

#why we use from_logits=True
#The Risky Way: If you calculate Softmax first, you might get a tiny probability like $0.000000000001$. 
# When the Loss Function then tries to take the log() of that tiny number, the computer can get confused
#  and return NaN (Not a Number) or Inf (Infinity). This effectively "breaks" your training.
preferred_model = tf.keras.models.Sequential(
    [ 
        tf.keras.layers.Dense(25, activation = 'relu'),
        tf.keras.layers.Dense(15, activation = 'relu'),
        tf.keras.layers.Dense(4, activation = 'linear')   #<-- Note
    ]
)
preferred_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  #<-- Note
    optimizer=tf.keras.optimizers.Adam(0.001),
)

preferred_model.fit(
    X_train,y_train,
    epochs=10
)

pre=preferred_model.predict(X_train)

print(pre[:2])

sm_pre = tf.nn.softmax(pre).numpy()

for i in range(5):
    print(f"actual class:{y_train[i]}  pedicted class:{np.argmax(sm_pre[i])}")