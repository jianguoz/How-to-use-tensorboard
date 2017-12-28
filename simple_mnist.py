import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01),name=name)

def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
    # Add layer name scopes for better graph visualization
    with tf.name_scope("layer1"):
        X = tf.nn.dropout(X, p_keep_input)
        h = tf.nn.relu(tf.matmul(X, w_h))
    with tf.name_scope("layer2"):
        h = tf.nn.dropout(h, p_keep_hidden)
        h2 = tf.nn.relu(tf.matmul(h, w_h2))
    with tf.name_scope("layer3"):
        h2 = tf.nn.dropout(h2, p_keep_hidden)
        h2 = tf.nn.relu(tf.matmul(h2, w_o))
        return h2

# Load data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# Create input and output placeholders for data
X = tf.placeholder(tf.float32, [None, 784], name = "input_X")
Y = tf.placeholder(tf.float32, [None, 10], name = "label_Y")

# Initialize weights
w_h = init_weights([784, 625], "w_h")
w_h2 = init_weights([625, 625], "w_h2")
w_o = init_weights([625, 10], "w_o")

# Add histogram for weights
tf.summary.histogram("w_h_summ", w_h)
tf.summary.histogram("w_h2_summ", w_h2)
tf.summary.histogram("w_o_summ", w_o)

# Add dropout
p_keep_input = tf.placeholder(tf.float32, name="p_keep_input")
p_keep_hidden = tf.placeholder(tf.float32, name="p_keep_hidden")

# Create model
logits = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

# Create cost function
with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    # Add scalar summary for cost tensor
    tf.summary.scalar("cost", cost)

# Measure accuracy
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # Add scalar summary for accuracy tensor
    tf.summary.scalar("accuracy", acc_op)

# Create session
with tf.Session() as sess:
    # Create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
    writer = tf.summary.FileWriter("./logs/nn_logs", sess.graph)
    merged = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())

    for i in range(15):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trY)+1, 128)):
            sess.run(train_op, feed_dict={X:trX[start:end], Y:trY[start:end],\
                                          p_keep_input:0.8, p_keep_hidden:0.5})
        summary, acc = sess.run([merged, acc_op], feed_dict={X:teX, Y:teY, \
                                                           p_keep_input:1.0,\
                                                           p_keep_hidden:1.0})
        # Write summary
        writer.add_summary(summary,i)
        print(i,acc)


# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import tensorflow as tf
#
# from tensorflow.examples.tutorials.mnist import input_data
#
# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
#                      'for unit testing.')
# flags.DEFINE_integer('max_steps', 101, 'Number of steps to run trainer.')
# flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
# flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
# flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
# flags.DEFINE_string('summaries_dir', '/tmp/mnist_logs', 'Summaries directory')
#
#
# def train():
#   # Import data
#   mnist = input_data.read_data_sets(FLAGS.data_dir,
#                                     one_hot=True,
#                                     fake_data=FLAGS.fake_data)
#
#   sess = tf.InteractiveSession()
#
#   # Create a multilayer model.
#
#   # Input placeholders
#   with tf.name_scope('input'):
#     x = tf.placeholder(tf.float32, [None, 784], name='x-input')
#     y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
#
#   with tf.name_scope('input_reshape'):
#     image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
#     tf.summary.image('input', image_shaped_input, 10)
#
#   # We can't initialize these variables to 0 - the network will get stuck.
#   def weight_variable(shape):
#     """Create a weight variable with appropriate initialization."""
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
#   def bias_variable(shape):
#     """Create a bias variable with appropriate initialization."""
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
#
#   def variable_summaries(var, name):
#     """Attach a lot of summaries to a Tensor."""
#     with tf.name_scope('summaries'):
#       mean = tf.reduce_mean(var)
#       tf.summary.scalar('mean/' + name, mean)
#       with tf.name_scope('stddev'):
#         stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#       tf.summary.scalar('stddev/' + name, stddev)
#       tf.summary.scalar('max/' + name, tf.reduce_max(var))
#       tf.summary.scalar('min/' + name, tf.reduce_min(var))
#       tf.summary.histogram(name, var)
#
#   def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
#     """Reusable code for making a simple neural net layer.
#     It does a matrix multiply, bias add, and then uses relu to nonlinearize.
#     It also sets up name scoping so that the resultant graph is easy to read,
#     and adds a number of summary ops.
#     """
#     # Adding a name scope ensures logical grouping of the layers in the graph.
#     with tf.name_scope(layer_name):
#       # This Variable will hold the state of the weights for the layer
#       with tf.name_scope('weights'):
#         weights = weight_variable([input_dim, output_dim])
#         variable_summaries(weights, layer_name + '/weights')
#       with tf.name_scope('biases'):
#         biases = bias_variable([output_dim])
#         variable_summaries(biases, layer_name + '/biases')
#       with tf.name_scope('Wx_plus_b'):
#         preactivate = tf.matmul(input_tensor, weights) + biases
#         tf.summary.histogram(layer_name + '/pre_activations', preactivate)
#       activations = act(preactivate, name='activation')
#       tf.summary.histogram(layer_name + '/activations', activations)
#       return activations
#
#   hidden1 = nn_layer(x, 784, 500, 'layer1')
#
#   with tf.name_scope('dropout'):
#     keep_prob = tf.placeholder(tf.float32)
#     tf.summary.scalar('dropout_keep_probability', keep_prob)
#     dropped = tf.nn.dropout(hidden1, keep_prob)
#
#   y = nn_layer(dropped, 500, 10, 'layer2', act=tf.nn.softmax)
#
#   with tf.name_scope('cross_entropy'):
#     diff = y_ * tf.log(y)
#     with tf.name_scope('total'):
#       cross_entropy = -tf.reduce_mean(diff)
#     tf.summary.scalar('cross entropy', cross_entropy)
#
#   with tf.name_scope('train'):
#     train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
#         cross_entropy)
#
#   with tf.name_scope('accuracy'):
#     with tf.name_scope('correct_prediction'):
#       correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#     with tf.name_scope('accuracy'):
#       accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     tf.summary.scalar('accuracy', accuracy)
#
#   # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
#   #merged = tf.merge_all_summaries()
#   merged = tf.summary.merge_all()
#   train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
#                                         sess.graph)
#   test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
#   tf.initialize_all_variables().run()
#
#   # Train the model, and also write summaries.
#   # Every 10th step, measure test-set accuracy, and write test summaries
#   # All other steps, run train_step on training data, & add training summaries
#
#   def feed_dict(train):
#     """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
#     if train or FLAGS.fake_data:
#       xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
#       k = FLAGS.dropout
#     else:
#       xs, ys = mnist.test.images, mnist.test.labels
#       k = 1.0
#     return {x: xs, y_: ys, keep_prob: k}
#
#   for i in range(FLAGS.max_steps):
#     if i % 10 == 0:  # Record summaries and test-set accuracy
#       summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
#       test_writer.add_summary(summary, i)
#       print('Accuracy at step %s: %s' % (i, acc))
#     else:  # Record train set summaries, and train
#       if i % 100 == 99:  # Record execution stats
#         run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#         run_metadata = tf.RunMetadata()
#         summary, _ = sess.run([merged, train_step],
#                               feed_dict=feed_dict(True),
#                               options=run_options,
#                               run_metadata=run_metadata)
#         train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
#         train_writer.add_summary(summary, i)
#         print('Adding run metadata for', i)
#       else:  # Record a summary
#         summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
#         train_writer.add_summary(summary, i)
#   train_writer.close()
#   test_writer.close()
#
#
# def main(_):
#   if tf.gfile.Exists(FLAGS.summaries_dir):
#     tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
#   tf.gfile.MakeDirs(FLAGS.summaries_dir)
#   train()
#
#
# if __name__ == '__main__':
#   tf.app.run()
