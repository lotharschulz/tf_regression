import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os


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
eta = 0.01

theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name='theta')
y_prediction = tf.matmul(X, theta, name='prediction')

with tf.name_scope('loss') as scope:
    error = y_prediction - y
    mse = tf.reduce_mean(tf.square(error), name='mse')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=eta)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()

saver = tf.train.Saver()

mse_summary = tf.summary.scalar('MSE', mse)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

with tf.Session() as sess:
    sess.run(init)
    if os.path.isdir(os.path.join(checkpoints_dir, checkpoints_file)):
        saver.restore(sess,os.path.join(checkpoints_dir, checkpoints_file))
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    save_path = saver.save(sess, os.path.join(checkpoints_dir, checkpoints_file))