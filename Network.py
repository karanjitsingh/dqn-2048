import tensorflow as tf
import numpy as np

tf.reset_default_graph()


def new_FCN(input_size, hidden_layers, output_size):
	last_layer = input_size
	weights = []
	biases = []

	inputs1 = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
	activation = [inputs1]

	for i, n in enumerate(hidden_layers):
		weights.append(tf.Variable(tf.truncated_normal([last_layer, n], 0, 0.1)))
		biases.append(tf.Variable(tf.random_normal(stddev=0.1, shape=[n])))
		activation.append(tf.nn.relu(tf.add(tf.matmul(activation[i], weights[i]), biases[i])))
		last_layer = n

	weights.append(tf.Variable(tf.truncated_normal([last_layer, output_size], 0, 0.1)))
	biases.append(tf.Variable(tf.random_normal(stddev=0.1, shape=[n])))

	return tf.nn.relu(tf.add(tf.matmul(activation[-1], weights[-1]), biases[-1])), inputs1


def new_CNN(input_size, output_size):
	def flatten_layer(layer):
		layer_shape = layer.get_shape()
		num_features = np.array(layer_shape[1:4], dtype=int).prod()
		layer_flat = tf.reshape(layer, [-1, num_features])
		return layer_flat, num_features

	def new_fc_layer(input,
					 num_inputs,
					 num_outputs,
					 use_relu=True):
		weights = tf.Variable(tf.truncated_normal(shape=[num_inputs, num_outputs], stddev=0.1))
		biases = tf.Variable(tf.constant(0.1, shape=[num_outputs]))

		layer = tf.matmul(input, weights) + biases

		if use_relu:
			layer = tf.nn.relu(layer)

		return layer

	def new_conv_layer(inputs, filter_size, num_filters, prev_filters=1, maxpool=False):
		shape = [filter_size, filter_size, prev_filters, num_filters]
		W = tf.Variable(tf.truncated_normal(shape, 0, 0.1))
		b = tf.Variable(tf.random_normal(stddev=0.1, shape=[num_filters]))

		layer = tf.nn.conv2d(
			input=inputs,
			filter=W,
			strides=[1, 1, 1, 1],
			padding='VALID') + b

		if maxpool:
			layer = tf.nn.max_pool(
				value=layer,
				ksize=[1, 2, 2, 1],
				strides=[1, 1, 1, 1],
				padding='VALID')

		return layer

	inputs = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
	inputs_2d = tf.reshape(inputs, [-1, 4, 4, 1])

	layer = new_conv_layer(inputs_2d, 2, 256)
	layer = new_conv_layer(layer, 2, 128, prev_filters=256)

	layer, features = flatten_layer(layer)

	fc = new_fc_layer(layer, features, 256)
	fc = new_fc_layer(fc, 256, 4)

	return fc, inputs


def getNetworkFromArgs(arg):
	if arg[0] == "fcn":
		outputs, inputs = new_FCN(16, arg[1], 4)
		return outputs, inputs
	if arg[0] == "cnn":
		outputs, inputs = new_CNN(16, 4)
		return outputs, inputs
