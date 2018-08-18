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

# optimizers from https://www.tensorflow.org/api_guides/python/train#optimizers
# see also https://www.tensorflow.org/api_docs/python/tf/contrib/opt
gradient_descent_training_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(mse)
adelta_training_op = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(mse)
adagrad_training_op = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(mse)

global_step = variables.Variable(0, dtype=dtypes.int64)
adagradDA_training_op = tf.train.AdagradDAOptimizer(learning_rate=learning_rate,
                                                    global_step=global_step,
                                                    initial_gradient_squared_accumulator_value=0.1,
                                                    l1_regularization_strength=0.0,
                                                    l2_regularization_strength=0.0).minimize(mse)

momentum_training_op = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(mse)
adam_training_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mse)
ftrl_training_op = tf.train.FtrlOptimizer(learning_rate=learning_rate).minimize(mse)
proximal_gradient_descent_training_op = \
    tf.train.ProximalGradientDescentOptimizer(learning_rate=learning_rate).minimize(mse)
proximal_adagrad_training_op = tf.train.ProximalAdagradOptimizer(learning_rate=learning_rate).minimize(mse)
rmsprop_training_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(mse)

init = tf.global_variables_initializer()

run_session(init, "gradient descent optimizer", gradient_descent_training_op, n_epochs, theta)
run_session(init, "adelta optimizer", adelta_training_op, n_epochs, theta)
run_session(init, "adagrad optimizer", adagrad_training_op, n_epochs, theta)
run_session(init, "adagrad dual averaging optimizer", adagradDA_training_op, n_epochs, theta)
run_session(init, "momentum optimizer", momentum_training_op, n_epochs, theta)
run_session(init, "adam optimizer", adam_training_op, n_epochs, theta)
run_session(init, "FTRL algorithm optimizer", ftrl_training_op, n_epochs, theta)
run_session(init, "proximal gradient descent optimizer", proximal_gradient_descent_training_op, n_epochs, theta)
run_session(init, "proximal adagrad optimizer", proximal_adagrad_training_op, n_epochs, theta)
run_session(init, "RMSProp algorithm optimizer", rmsprop_training_op, n_epochs, theta)

