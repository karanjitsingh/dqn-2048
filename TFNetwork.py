import tensorflow as tf
import numpy as np

tf.reset_default_graph()


def new_FCN(input_size, hidden_layers, output_size):
	last_layer = 16
	weights = []
	biases = []

	inputs1 = tf.placeholder(shape=[1, input_size], dtype=tf.float32)
	activation = [inputs1]

	for i, n in enumerate(hidden_layers):
		weights.append(tf.Variable(tf.truncated_normal([last_layer, n], 0, 0.1)))
		biases.append(tf.Variable(tf.constant(0.1, shape=[n])))
		activation.append(tf.nn.relu(tf.add(tf.matmul(activation[i], weights[i]), biases[i])))
		last_layer = n

	weights.append(tf.Variable(tf.truncated_normal([last_layer, output_size], 0, 0.1)))
	biases.append(tf.Variable(tf.constant(0.1, shape=[output_size])))

	return tf.nn.relu(tf.add(tf.matmul(activation[-1], weights[-1]), biases[-1]))


def new_Conv(input_size, output_size):

	inputs1 = tf.placeholder(shape=[1, input_size], dtype=tf.float32)

	def new_weights(shape):
		return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

	def new_biases(length):
		return tf.Variable(tf.constant(0.1, shape=[length]))

	def new_conf_layer(input, filter_size, num_filters):
		shape = [filter_size, filter_size,1,num_filters]
		weights = new_weights(shape)
		biases = new_biases(length=num_filters)

		layer=tf.nn.conv2d(input=input,
						   filter=weights,
						   strides=[1,1,1,1],
						   padding='SAME') + biases

		layer = tf.nn.max_pool(value=layer,
							   ksize=[1,2,2,1],
							   strides=[1,2,2,1],
							   padding='SAME')

		layer = tf.nn.sigmoid(layer)

		return layer, weights

	def new_fc_layer(input,
					 num_inputs,
					 num_outputs,
					 use_relu=True):

		weights = new_weights(shape=[num_inputs, num_outputs])
		biases = new_biases(length=num_outputs)

		layer = tf.matmul(input, weights) + biases

		if use_relu:
			layer = tf.nn.relu(layer)

		return layer, inputs1

	def flatten_layer(layer):

		layer_shape = layer.get_shape()
		num_features = np.array(layer_shape[1:4], dtype=int).prod()
		layer_flat = tf.reshape(layer, [-1, num_features])

		return layer_flat, num_features

	inputs_2d= tf.reshape(inputs1, [-1, 4, 4, 1])

	layer_1, _ = new_conf_layer(inputs_2d, filter_size=2, num_filters=9)

	layer_flat, num_features = flatten_layer(layer_1)

	inputs_2d = tf.reshape(layer_flat, [-1, 6, 6, 1])

	layer_2, _ = new_conf_layer(inputs_2d, filter_size=2, num_filters = 4)

	layer_flat, num_features = flatten_layer(layer_2)

	layer_fc1, _ = new_fc_layer(input = layer_flat,
							 num_inputs= num_features,
							 num_outputs= 36)

	layer_fc2, _ = new_fc_layer(input= layer_fc1,
							 num_inputs= 36,
							 num_outputs= output_size)

	return layer_fc2, inputs1
