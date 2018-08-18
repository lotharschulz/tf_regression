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

scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data.data]

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

theta = tf.Variable(tf.random_uniform([n+1,1], -1.0,1.0), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')

error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')

gradients = tf.gradients(mse, [theta])[0]
gradient_descent_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
momentum_optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

gradient_descent_training_op = gradient_descent_optimizer.minimize(mse)
momentum_training_op = momentum_optimizer.minimize(mse)

init = tf.global_variables_initializer()

with tf.Session() as gradient_descent_sess:
    gradient_descent_sess.run(init)
    print("gradient descent optimizer")
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        gradient_descent_sess.run(gradient_descent_training_op)
    best_theta = theta.eval()
    print("best theta\n", best_theta)

with tf.Session() as momentum_sess:
    momentum_sess.run(init)
    print("momentum optimizer")
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        momentum_sess.run(momentum_training_op)
    best_theta = theta.eval()
    print("best theta\n", best_theta)

# gradient descent optimizer
# Epoch 0 MSE = 8.25434
# Epoch 100 MSE = 0.948461
# Epoch 200 MSE = 0.748185
# Epoch 300 MSE = 0.68909
# Epoch 400 MSE = 0.647255
# Epoch 500 MSE = 0.616427
# Epoch 600 MSE = 0.59362
# Epoch 700 MSE = 0.576691
# Epoch 800 MSE = 0.564082
# Epoch 900 MSE = 0.554657
# best theta
#  [[ 2.06855249]
#  [ 0.87705243]
#  [ 0.17237075]
#  [-0.27149037]
#  [ 0.27383834]
#  [ 0.01438697]
#  [-0.04487337]
#  [-0.46447456]
#  [-0.43660375]]
# momentum optimizer
# Epoch 0 MSE = 15.1179
# Epoch 100 MSE = 0.530413
# Epoch 200 MSE = 0.524698
# Epoch 300 MSE = 0.524364
# Epoch 400 MSE = 0.524326
# Epoch 500 MSE = 0.524322
# Epoch 600 MSE = 0.524321
# Epoch 700 MSE = 0.524321
# Epoch 800 MSE = 0.524321
# Epoch 900 MSE = 0.524321
# best theta
#  [[ 2.06855798]
#  [ 0.82962567]
#  [ 0.11875275]
#  [-0.26553884]
#  [ 0.30570623]
#  [-0.00450266]
#  [-0.0393265 ]
#  [-0.89987195]
#  [-0.87052804]]

