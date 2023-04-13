import tensorflow as tf

# define model architecture and loss function
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.build(input_shape=(None, 128))
model.summary()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# define optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# define inputs and targets
inputs = tf.random.normal((32, 128))
targets = tf.random.uniform((32,), maxval=10, dtype=tf.int32)

# use GradientTape context to record the operations
with tf.GradientTape() as tape:
    # execute forward pass
    predictions = model(inputs)
    # calculate loss
    loss = loss_fn(targets, predictions)
# calculate gradients
gradients = tape.gradient(loss, model.trainable_variables)

# apply gradients to model parameters using optimizer
optimizer.apply_gradients(zip(gradients, model.trainable_variables))

import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
print(f'a.shape: {a.shape}')
b = np.array([[1, 2], [3, 4], [5, 6]])
print(f'b.shape: {b.shape}')
c = np.dot(a, b)
print(c)
d = np.matmul(a, b)
print(d)

print('Done!')