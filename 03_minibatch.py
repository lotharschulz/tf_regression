import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from tensorflow.python.ops import variables
from tensorflow.python.framework import dtypes
from datetime import datetime
import os


def run_session(label, op, initializer, checkpoints_directory, check_points_file, epochs, batches, batchsize,
                mse_sum, theta2):
    with tf.Session() as sess:
        sess.run(initializer)
        print("############- ", label, " -#########")
        if os.path.isdir(os.path.join(checkpoints_directory, check_points_file)):
            saver.restore(sess, os.path.join(checkpoints_directory, check_points_file))
        for epoch in range(epochs):
            for batch_index in range(batches):
                X_batch, y_batch = fetch_batch(batch_index, batchsize)
                if batch_index % 10 == 0:
                    summary_str = mse_sum.eval(feed_dict={X: X_batch, y: y_batch})
                    step = epoch * batches + batch_index
                    file_writer.add_summary(summary_str, step)
                sess.run(op, feed_dict={X: X_batch, y: y_batch})

        best_theta = theta2.eval()
        print("best theta\n", best_theta)
        if not os.path.isdir(checkpoints_directory):
            os.makedirs(checkpoints_directory)
        save_path = saver.save(sess, os.path.join(checkpoints_directory, check_points_file))


def fetch_batch(batchindex, batchsize):
    start = batchindex * batchsize
    if batchindex == 206:
        Xbatch = scaled_housing_data_plus_bias[start:, :]
        ybatch = housing.target.reshape(-1, 1)[start:, :]
    else:
        Xbatch = scaled_housing_data_plus_bias[start:start + batchsize, :]
        ybatch = target[start:start + batchsize, :]
    return Xbatch, ybatch


checkpoints_dir = 'out/tf_tmp_checkpoints/'
checkpoints_file = 'model.ckpt'
root_log_dir = 'out/tf_log'
now = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
log_dir = '{}/run-{}/'.format(root_log_dir, now)

housing = fetch_california_housing()
m, n = housing.data.shape

scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data.astype(np.float32))
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
target = housing.target.reshape(-1, 1)

X = tf.placeholder(dtype=tf.float32, shape=(None, n+1), name='X')
y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='y')

n_epochs = 50
batch_size = 100
n_batches = int(np.ceil(m/batch_size))
learning_rate = 0.01

theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name='theta')
y_prediction = tf.matmul(X, theta, name='prediction')

with tf.name_scope('loss') as scope:
    error = y_prediction - y
    mse = tf.reduce_mean(tf.square(error), name='mse')

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=eta)

ops = list()

# optimizers: https://www.tensorflow.org/api_guides/python/train#optimizers
# see also https://www.tensorflow.org/api_docs/python/tf/contrib/opt
ops.append(("gradient descent optimizer",
            tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(mse)))
ops.append(("adelta optimizer",
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


# training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()

saver = tf.train.Saver()

mse_summary = tf.summary.scalar('MSE', mse)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

for o in ops:
    run_session(o[0], o[1], init, checkpoints_dir, checkpoints_file, n_epochs,
                n_batches, batch_size, mse_summary, theta)
