from subprocess import Popen
import tensorflow as tf
import webbrowser
import shutil
import sys
import os

if sys.gettrace() is None:
	log_path = "./log"
else:
	log_path = "./debug"

print log_path

writer = None
model_id = ''


def init_summary_writer(model_name, var_list, tb_port):
	with tf.name_scope(model_name):
		for tuple in var_list:
			tf.summary.scalar(tuple[0], tuple[1])

	if os.path.isdir(log_path):
		try:
			shutil.rmtree(log_path)
		except:
			exit()
			pass
	else:
		while not os.path.isdir(log_path):
			os.makedirs(log_path)

	global writer
	global model_id

	model_id = model_name
	writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())

	print os.path.abspath(log_path)

	if tb_port:
		Popen(["tensorboard", "--logdir=" + os.path.abspath(log_path), "--port="+tb_port])
		webbrowser.open("http://localhost:"+tb_port)


	return tf.summary.merge_all()


def write_scalar_summaries(summary_list, step):
	if writer is None:
		print "Summary writer not initialized."
		return

	for t in summary_list:
		s = tf.Summary(value=[
			tf.Summary.Value(tag=str(model_id + "/" + t[0]), simple_value=t[1]),
		])

		writer.add_summary(s, step)


def write_summary_operation(summary, step):
	if writer is None:
		print "Summary writer not initialized."
		return

	writer.add_summary(summary, step)