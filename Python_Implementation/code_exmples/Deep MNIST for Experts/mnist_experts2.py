'''
This extends from mnist_experts1.py to a Multilayer Convolutional Network
'''
#Tutorial: https://www.tensorflow.org/get_started/mnist/pros

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

#One should generally initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#Since we're using ReLU neurons, it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons"
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#convolutions uses a stride of one and are zero padded so that the output is the same size as the input
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#Our pooling is plain old max pooling over 2x2 blocks
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#main
#TODO: Maybe delete
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#The convolution will compute 32 features for each 5x5 patch
#The first two dimensions are the patch size, the next is the number of input channels, and the last is the number of output channels.
#In other words 32 filters each one of size 5x5. The input has only one channel (instead of for example 3 if the images were RGB).
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#reshaping the image
x_image = tf.reshape(x, [-1,28,28,1])

#We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, 
#and finally max pool. The max_pool_2x2 method will reduce the image size to 14x14.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Second convolutional layer
#The second layer will have 64 features for each 5x5 patch. This will reduce the image to a image size has been reduced to 7x7
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Densely Connected Layer
#Now that the image size has been reduced to 7x7, we add a fully-connected layer with 1024 neurons to allow
#processing on the entire image. We reshape the tensor from the pooling layer into a batch of vectors, 
#multiply by a weight matrix, add a bias, and apply a ReLU.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout Layer
#To reduce overfitting, we will apply dropout before the readout layer. We create a placeholder for the
#probability that a neuron's output is kept during dropout. This allows us to turn dropout on during training,
#and turn it off during testing. TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in
#addition to masking them, so dropout just works without any additional scaling.1
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Train and Evaluate the Model
#To train and evaluate it we will use code that is nearly identical to that for the simple one layer SoftMax 
#network above. The differences are that:
#1. We will replace the steepest gradient descent optimizer with the more sophisticated ADAM optimizer.
#2. We will include the additional parameter keep_prob in feed_dict to control the dropout rate.
#3. We will add logging to every 100th iteration in the training process.

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
	batch = mnist.train.next_batch(50)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))
	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

