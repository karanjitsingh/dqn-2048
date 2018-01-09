import tensorflow as tf
import shutil

log_path = "./log"
writer = None
trainer_id = ''


def init_summary_writer(training_id, var_list):
	with tf.name_scope(training_id):
		for tuple in var_list:
			tf.summary.scalar(tuple[0], tuple[1])

	try:
		shutil.rmtree(log_path)
	except:
		pass

	global writer
	global trainer_id

	trainer_id = training_id
	writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())

	return tf.summary.merge_all()


def write_scalar_summaries(summary_list, step):
	if writer is None:
		print "Summary writer not initialized."
		return

	for t in summary_list:
		s = tf.Summary(value=[
			tf.Summary.Value(tag=str(trainer_id + "/" + t[0]), simple_value=t[1]),
		])

		writer.add_summary(s, step)


def write_summary_operation(summary, step):
	if writer is None:
		print "Summary writer not initialized."
		return

	writer.add_summary(summary, step)