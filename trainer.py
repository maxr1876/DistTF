import socket
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
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

'''
These four functions help to define the CNN model to be trained. 
Instead of using just a simple softmax function with a neural net, this code will 
implement a multilayer convolutional neural network. In order to create this model,
we will need many weights and biases. Since we're  using ReLU (Rectified Linear Unit) 
neurons, it is also good practice to initialize them with a slightly positive 
initial bias to avoid "dead neurons". (https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks)
Instead of doing this repeatedly while we build the model, weight_variable() and bias_variable()
can be called to return a variable or constant based on the shape provided.
'''
def weight_variable(shape):
  '''
	tf.truncated_normal returns random values from a truncated normal distribution. 
	The generated values follow a normal distribution with specified mean and standard 
	deviation, except that values whose magnitude is more than 2 standard deviations
	from the mean are dropped and re-picked.
	'''
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

'''
These next two functions perform convolution and pooling. Convolution is the process of training on portions of 
an image, and applying the features learned from that portion to the entire image. The stride indicates 
how many pixels to shift over when applying this 'mask' to the entire image. In our case, we use the 
default of 1. Pooling is a sample based discretization process. The objective is to down-sample the input 
into bins.
'''
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  
  '''
  After creating a list of param servers and workers, create an instance 
  of tf.train.ClusterSpec. This defines what the cluster looks like, so 
  each machines are aware of both itelf and all the other machines.
  '''
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  '''
  In order to communicate with the other machines, a server must be created.
  It is important to provide the server with the ClusterSpec object as well
  as the name of the current job along with the task_index of the job. This
  way, the server is aware of each machine's role within the system.
  '''
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)
  '''
  Based on the job assigned to a machine, one of two things should happen.
  If the machine is a parameter server (ps), we call server.join(). This 
  blocks until the server is manually shut down. The reason for this is
  because the parameter server stores and updates all tf.Variable objects,
  but that is all handled automatically. All we need to do is let it
  do its thing until we tell it to be done.
  Otherwise, if the machine is a worker, we need to define all of the variables
  and operations that define our model.
  '''
  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    '''
    We must specify which device will take care of which operations. We have a
    two options. We can either manually specify which device will handle specific
    operations, or use a tf.train.replica_device_setter, which automatically handles
    assigning tasks to devices. All tf.Variable() objects will be stored on 'ps' 
    devices, while computational tasks (operations) will be placed on 'worker'
    devices. 
    '''
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/replica:0/task:%d" % FLAGS.task_index,
        cluster=cluster)):
      # global_step coordinates between all workers the current step of training
      global_step = tf.Variable(0, trainable=False)
      #More to come on is_chief...
      is_chief = FLAGS.task_index == 0
      
      '''
      A placeholder is a symbolic variable. It can be used as input to a 
      TensorFlow computation. The size 784 is due to the fact that all 
      input images are 28x28 pixels. Here None means that a dimension 
      can be of any length.
      '''
      x = tf.placeholder(tf.float32, [None, 784])
      y_ = tf.placeholder(tf.float32, [None, 10])
      
      #Read the input data
      mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
      
      '''
      Here we initialize the weights and bias. Create a tensorFlow Variable, which
      represents a modifiable Tensor that will be passed from node to node in the 
      computational graph. W has a shape of [784, 10] because we want to multiply
      the 784-dimensional image vectors by it to produce 10-dimensional vectors 
      of evidence for the difference classes. b has a shape of [10] so it can be 
      added to the output.
      '''
      W = tf.Variable(tf.zeros([784, 10]))
      b = tf.Variable(tf.zeros([10]))
      
      '''
      At this point, we can begin to define the model. We define y to be the softmax 
      function. softmax accepts a computation that will be performed at each node of
      the graph.. In our case, this is multiplying x*W, and adding b to the result 
      of that multiplication.
      '''
      y = tf.nn.softmax(tf.matmul(x, W) + b)
      
      '''
      Initialize the first convolutional layer. The convolution will compute 32 features for each 5x5 
      image patch. Hence the first two parameters being 5. The 1 represents the number of channels in the image.
      As these are grayscale images, there is only one channel. Were these to be color images, there would be 3
      channels (rgb). The fourth argument is the number of channels to output. We will also need a bias vector
      with a component for each output channel.
      '''
      W_conv1 = weight_variable([5, 5, 1, 32])
      b_conv1 = bias_variable([32])
      
      '''
      To apply the layer, we first reshape x to a 4d tensor, with the second and third dimensions corresponding 
      to image width and height, and the final dimension corresponding to the number of color channels.
      '''
      x_image = tf.reshape(x, [-1, 28, 28, 1])
      
      '''
      We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool. 
      The max_pool_2x2 method will reduce the image size to 14x14.
      '''
      h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
      h_pool1 = max_pool_2x2(h_conv1)
      
      '''
      Now initialize the second convolutional layer. This layer will be more dense, computing 64 features instead of 32.
      '''
      W_conv2 = weight_variable([5, 5, 32, 64])     
      b_conv2 = bias_variable([64])    
      h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
      h_pool2 = max_pool_2x2(h_conv2)
      
      '''
      Now it's time for the densely connected layer. This layer will have 1024 neurons to allow processing on
      an entire image. The images will now be 7x7 (caused by h_pool2).
      '''
      W_fc1 = weight_variable([7 * 7 * 64, 1024])
      b_fc1 = bias_variable([1024])
      h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
      
      '''
      To reduce overfitting, we will apply dropout before the readout layer. This allows units to be dropped, to avoid
      co-adaptations on training data. 
      '''
      keep_prob = tf.placeholder(tf.float32)
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
      
      '''
      Now add the final layer (readout layer). This is somewhat similar to the simple softmax example.
      (https://www.tensorflow.org/get_started/mnist/beginners)
      '''
      W_fc2 = weight_variable([1024, 10])
      b_fc2 = bias_variable([10])
      y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
      
      '''
      Now define the cross_entropy function, training step, correct prediction, and accuracy functions.
      When defining the train_step function, it is very important to include the global_step argument,
      otherwise the global step will never be incremented!
      '''
      cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
      train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy, global_step=global_step)
      correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      
      #saver allows for saving/restoring variables to/from checkpoints during training
      saver = tf.train.Saver()
      #summary_op tracks all summaries of the graph
      summary_op = tf.summary.merge_all()
      #init_op defines the operation to initialize all tf.Variable()s
      init_op = tf.global_variables_initializer()
      
      '''
      Create some configurations for the session below. In this case, not all values are needed,
      but can be used for specific configurations.
        allow_soft_placement=True: allows computations to be placed on devices that are not explicitly defined
        log_device_placement=False: don't log device placement...
        device_filters: only use devices whose names match the names provided here
      '''
      sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=["/job:ps", "/job:worker/replica:0/task:%d" % FLAGS.task_index])

      if is_chief:
        print("Worker %d: Initializing session..." % FLAGS.task_index)
      else:
        print("Worker %d: Waiting for session to be initialized..." %
            FLAGS.task_index)
    '''
    Create a "supervisor", which oversees the training process. The supervisor needs to know
    whether or not it is the chief (this is very important). There can be ONLY ONE chief. The
    chief is responsible for initializing the model. Until it is initialized, the other workers
    cannot continue. 
    '''
    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir="train_logs",
                             summary_op=summary_op,
                             init_op=init_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600)  
    server_grpc_url = "grpc://" + worker_hosts[FLAGS.task_index]

    '''
    Now it's time to initialize the session. Use sv.prepare_or_wait_for_session,
    so if a worker is the chief, it will initialize all that needs to be initialized,
    else it will wait until the chief has initialized everything. The first argument
    can be either the literal grpc url defined above, or 'server.target'.
    '''
    with sv.prepare_or_wait_for_session(server_grpc_url,
                                          config=sess_config) as sess:
      step = 0
      #If anything goes wrong, sv.should_stop() will halt execution on a worker
      while (not sv.should_stop()) and (step < 5000):
        # Run a training step asynchronously.
        batch = mnist.train.next_batch(FLAGS.batch_size)
        if step % 10 == 0:
          train_accuracy = accuracy.eval(session=sess,feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
          print('step %d, training accuracy %g' % (step, train_accuracy))
        _, step = sess.run([train_step, global_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

      print('test accuracy %g' % accuracy.eval(feed_dict={
          x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
      
    print("ALL FINISHED")
    sv.stop()



if __name__ == "__main__":
  tf.app.run()
