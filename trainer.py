import socket
import tensorflow as tf

'''
tf.app.flags is essentially a wrapper for Python's argparse. The values filled
in below are the default values, and can be changed with command line arguments.
For example, if you wanted to set the number of hidden units to 50, the appropriate
call would be "python3 trainer.py --hidden_units 50". 

In order to run TensorFlow in a distributed fashion, you must define the machines
in your cluster. You are required to have at least one parameter server, and at
least one worker. In this case, there is one parameter server and three workers.
If using more than one param server or worker, ensure that there are no spaces
between the names in the list (this caused a huge headache for me)!
'''

tf.app.flags.DEFINE_string("ps_hosts", "denver:42069",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "albany:42069,boston:42069,raleigh:42069",
                           "Comma-separated list of hostname:port pairs")



'''
This portion of code will vary greatly from user to user based on cluster setup.
I hard-coded in these machine names simply so I wouldn't have to pass them as
command line args each time. This code is defining the job of each machine.
There are two possible types of job: 'ps' and 'worker'. Within a job, each 
machine is assigned different portions of work based on its task number.
Since there is only one parameter server, its task_index within the job 'ps'
is 0. As there are three machines with the job 'worker', they are assigned 
task_index 0, 1, and 2. 
'''
name = socket.gethostname()
# Flags for defining the tf.train.Server
if name == 'denver':
  tf.app.flags.DEFINE_string("job_name", "ps", "One of 'ps', 'worker'")
  tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
elif name == 'albany':
  tf.app.flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker'")
  tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
elif name == 'boston':
  tf.app.flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker'")
  tf.app.flags.DEFINE_integer("task_index", 1, "Index of task within the job")
else:# name == raleigh
  tf.app.flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker'")
  tf.app.flags.DEFINE_integer("task_index", 2, "Index of task within the job")

tf.app.flags.DEFINE_integer("hidden_units", 100,
                            "Number of units in the hidden layer of the NN")
tf.app.flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                           "Directory for storing mnist data")
tf.app.flags.DEFINE_integer("batch_size", 100, "Training batch size")

FLAGS = tf.app.flags.FLAGS
#Define the size of the training images
IMAGE_PIXELS=28

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def simple_setter(ps_device="/job:ps/replica:0/task:0"):
  def _assign(op):
    node_def = op if isinstance(op, tf.NodeDef) else op.node_def
    if node_def.op == "Variable":
        return ps_device
    else:
      return "/job:worker/replica:0/task:%d" % (FLAGS.task_index)
  return _assign


def main(_):
  from tensorflow.examples.tutorials.mnist import input_data
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)
  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/replica:0/task:%d" % FLAGS.task_index,
        cluster=cluster)):
    # with tf.device(simple_setter()):
      # Build model...
      global_step = tf.Variable(0, trainable=False)
      is_chief = FLAGS.task_index == 0
      # train_op = tf.train.AdagradOptimizer(0.01).minimize(
      #     loss, global_step=global_step)
      x = tf.placeholder(tf.float32, [None, 784])
      y_ = tf.placeholder(tf.float32, [None, 10])
      mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

      W = tf.Variable(tf.zeros([784, 10]))
      b = tf.Variable(tf.zeros([10]))

      y = tf.nn.softmax(tf.matmul(x, W) + b)

      W_conv1 = weight_variable([5, 5, 1, 32])
      b_conv1 = bias_variable([32])
      x_image = tf.reshape(x, [-1, 28, 28, 1])
      h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
      W_conv2 = weight_variable([5, 5, 32, 64])     
      b_conv2 = bias_variable([64])
      h_pool1 = max_pool_2x2(h_conv1)
      h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
      h_pool2 = max_pool_2x2(h_conv2)
      keep_prob = tf.placeholder(tf.float32)
      W_fc1 = weight_variable([7 * 7 * 64, 1024])
      b_fc1 = bias_variable([1024])
      h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)       
      W_fc2 = weight_variable([1024, 10])
      b_fc2 = bias_variable([10])
      y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
      cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

      # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
      train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy, global_step=global_step)
      correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      saver = tf.train.Saver()
      summary_op = tf.summary.merge_all()
      init_op = tf.global_variables_initializer()
      sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=["/job:ps", "/job:worker/replica:0/task:%d" % FLAGS.task_index])

      if is_chief:
        print("Worker %d: Initializing session..." % FLAGS.task_index)
      else:
        print("Worker %d: Waiting for session to be initialized..." %
            FLAGS.task_index)

      # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir="train_logs",
                             summary_op=summary_op,
                             init_op=init_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600)  
    server_grpc_url = "grpc://" + worker_hosts[FLAGS.task_index]

    
    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    # with sv.managed_session(server.target):
    with sv.prepare_or_wait_for_session(server_grpc_url,#first arg was previously server_grpc_url
                                          config=sess_config) as sess:
      step = 0
      local_step = 0
      while (not sv.should_stop()) and (step < 500):
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.
        batch = mnist.train.next_batch(FLAGS.batch_size)
        if step % 10 == 0:
          train_accuracy = accuracy.eval(session=sess,feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
          print('step %d, training accuracy %g' % (step, train_accuracy))
        _, step = sess.run([train_step, global_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        
        local_step += 1
      print('test accuracy %g' % accuracy.eval(feed_dict={
          x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
      
    print("ALL FINISHED")
    sv.stop()



if __name__ == "__main__":
  tf.app.run()
