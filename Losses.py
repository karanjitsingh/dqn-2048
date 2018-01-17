import tensorflow as tf


def mse(target, prediction):
	return tf.reduce_sum(tf.square(target - prediction))


def huber_loss(target, prediction):
	print target, prediction
	err = target - prediction
	return tf.where(tf.abs(err) < 1.0, 0.5 * tf.square(err), tf.abs(err) - 0.5)
