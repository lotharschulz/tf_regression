import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from datetime import datetime


# def fetch_batch(e_poch, nbatches, batchindex, batchsize, size):
#     np.random.seed(e_poch * nbatches + batchindex)
#     indices = np.random.randint(size, size=batchsize)
#     Xbatch = housing_data_plus_bias[indices]
#     ybatch = housing.target.reshape(-1, 1)[indices]
#     return Xbatch, ybatch
#     # m, _ = scaled_housing_data_plus_bias.shape
#     # if (batch_index+1)*batch_size > m:
#     #     to = m
#     # else:
#     #     to = (batch_index+1)*batch_size
#     # Xbatch = scaled_housing_data_plus_bias[batch_index*batch_size:to]
#     # ybatch = housing.target.reshape(-1, 1)[batch_index*batch_size:to]
#     # return Xbatch, ybatch


def fetch_batch(batchindex, batchsize):
    start = batchindex * batchsize
    if batchindex == 206:
        Xbatch = housing_data_plus_bias[start:, :]
        ybatch = housing.target.reshape(-1, 1)[start:, :]
    else:
        Xbatch = housing_data_plus_bias[start:start + batchsize, :]
        ybatch = target[start:start + batchsize, :]
    return Xbatch, ybatch


checkpoints_dir = '/home/lothar/Documents/NetBeansProjects/python/tf_regression/tmp/tf_tmp_checkpoints/'
root_log_dir = '/home/lothar/Documents/NetBeansProjects/python/tf_regression/tmp/tf_log'
now = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
log_dir = '{}/run-{}/'.format(root_log_dir, now)

housing = fetch_california_housing()
m, n = housing.data.shape

scaler = StandardScaler()
housing_scaled = scaler.fit_transform(housing.data.astype(np.float32))
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing_scaled]
target = housing.target.reshape(-1, 1)

X = tf.placeholder(dtype=tf.float32, shape=(None, n+1), name='X')
y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='y')


n_epochs = 50
batch_size = 100
n_batches = int(np.ceil(m/batch_size))
eta = 0.01

theta = tf.Variable(tf.random_uniform([n+1, 1], -1, 1), name='theta')
prediction = tf.matmul(X, theta, name='prediction')

with tf.name_scope('loss') as scope:
    error = prediction - y
    mse = tf.reduce_mean(tf.square(error), name='mse')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=eta)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()

saver = tf.train.Saver()

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

with tf.Session() as sess:
    sess.run(init)
    # saver.restore(sess, 'tf_tmp_checkpoints/mini_BatchGD.ckpt')
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(batch_index, batch_size)
            # X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()
    save_path = saver.save(sess, checkpoints_dir)

#############

#
#
# n_epochs = 1000
# learning_rate = 0.01
# batch_size = 100
#
# housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
# n_batches = int(np.ceil(m / batch_size))
#
# scaled_housing_data = scaler.fit_transform(housing.data.astype(np.float32))
# scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data.data]
#
# X = tf.placeholder(dtype=tf.float32, shape=(None, n+1), name='X')
# y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='y')
#
# theta = tf.Variable(tf.random_uniform([n+1,1], -1.0,1.0), name='theta')
# y_pred = tf.matmul(X, theta, name='predictions')
#
# error = y_pred - y
# mse = tf.reduce_mean(tf.square(error), name='mse')
#
# gradients = tf.gradients(mse, [theta])[0]
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# training_op = optimizer.minimize(mse)
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     for epoch in range(n_epochs):
#         for batch_index in range(n_batches):
#             X_batch, y_batch = fetch_batch(epoch, n_batches, batch_index, batch_size, m)
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#             #print("epoch", epoch, "batches", batch_index, " MSE=", mse.eval(feed_dict={X: X_batch, y: y_batch}))
#         if epoch % 100 == 0:
#             print("Epoch", epoch, "MSE =", mse.eval())
#     best_theta = theta.eval()