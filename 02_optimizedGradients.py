import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from tensorflow.python.ops import variables
from tensorflow.python.framework import dtypes


def run_session(initializer, label, op, epochs, theta_val):
    with tf.Session() as sess:
        sess.run(initializer)
        print("############- ", label, " -#########")
        for epoch in range(epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())
            sess.run(op)
        best_theta = theta_val.eval()
        print("best theta\n", best_theta)


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

ops = list()

# optimizers: https://www.tensorflow.org/api_guides/python/train#optimizers
# see also https://www.tensorflow.org/api_docs/python/tf/contrib/opt
ops.append(("gradient descent optimizer",
            tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(mse)))
ops.append(("adelta optimizer",
            tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(mse)))
ops.append(("adagrad optimizer",
            tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(mse)))
ops.append(("adagrad optimizer",
            tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(mse)))
ops.append(("adagrad dual averaging optimizer",
            tf.train.AdagradDAOptimizer(learning_rate=learning_rate,
                                        global_step=variables.Variable(0, dtype=dtypes.int64),
                                        initial_gradient_squared_accumulator_value=0.1,
                                        l1_regularization_strength=0.0,
                                        l2_regularization_strength=0.0).minimize(mse)))
ops.append(("momentum optimizer",
            tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(mse)))
ops.append(("adam optimizer",
            tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mse)))
ops.append(("FTRL algorithm optimizer",
            tf.train.FtrlOptimizer(learning_rate=learning_rate).minimize(mse)))
ops.append(("proximal gradient descent optimizer",
            tf.train.ProximalGradientDescentOptimizer(learning_rate=learning_rate).minimize(mse)))
ops.append(("proximal adagrad optimizer",
            tf.train.ProximalAdagradOptimizer(learning_rate=learning_rate).minimize(mse)))
ops.append(("RMSProp algorithm optimizer",
            tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(mse)))

init = tf.global_variables_initializer()

for t in ops:
    run_session(init, t[0], t[1], n_epochs, theta)
