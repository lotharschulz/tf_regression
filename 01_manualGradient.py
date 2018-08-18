import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


n_epochs = 1000
learning_rate = 0.01

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

# MSE starting between 14.1518 going to 0.534
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data.data]

# MSE starting between 5.83 going to 4.81
# housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
# scaled_housing_data_plus_bias = scaler.fit_transform(housing_data_plus_bias.data)

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

theta = tf.Variable(tf.random_uniform([n+1,1], -1.0,1.0), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')

error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')

# gradients = 2/m * tf.matmul(tf.transpose(X), error)
gradients = tf.gradients(mse, [theta])[0]
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()

# Epoch 0 MSE = 9.90542
# Epoch 100 MSE = 0.705636
# Epoch 200 MSE = 0.54114
# Epoch 300 MSE = 0.534413
# Epoch 400 MSE = 0.531796
# Epoch 500 MSE = 0.52993
# Epoch 600 MSE = 0.52855
# Epoch 700 MSE = 0.527523
# Epoch 800 MSE = 0.526758
# Epoch 900 MSE = 0.526185
