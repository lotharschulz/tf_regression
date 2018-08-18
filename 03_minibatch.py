import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

n_epochs = 1000
learning_rate = 0.01
batch_size = 100

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
n_batches = int(np.ceil(m / batch_size))


def fetch_batch(epoch, batch_index, batch_size):
    pass


scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data.data]

X = tf.placeholder(tf.float32, shape=(None, n+1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

theta = tf.Variable(tf.random_uniform([n+1,1], -1.0,1.0), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')

error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')

gradients = tf.gradients(mse, [theta])[0]
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# training_op = tf.assign(theta, theta - learning_rate * gradients)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            # hier weiter seite 240 fetch_batch(epoch, )
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()

# Epoch 0 MSE = 8.82812
# Epoch 100 MSE = 0.686054
# Epoch 200 MSE = 0.559898
# Epoch 300 MSE = 0.548698
# Epoch 400 MSE = 0.541957
# Epoch 500 MSE = 0.537106
# Epoch 600 MSE = 0.533597
# Epoch 700 MSE = 0.531057
# Epoch 800 MSE = 0.529217
# Epoch 900 MSE = 0.527884
