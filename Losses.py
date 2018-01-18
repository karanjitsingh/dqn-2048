import tensorflow as tf


def mse(target, prediction):
	return tf.reduce_sum(tf.square(target - prediction))


def huber_loss(target, prediction):
	return tf.losses.huber_loss(target, prediction)


def getLossFromArgs(args):
	if args == "mse":
		return mse
	if args == "huber":
		return huber_loss
