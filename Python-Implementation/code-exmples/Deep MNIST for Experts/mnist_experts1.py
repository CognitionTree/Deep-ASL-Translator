'''
This code builds a simple Softmax Regression Model
'''
#Tutorial: https://www.tensorflow.org/get_started/mnist/pros

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

#Build a Softmax Regression Model---------------------------------

#This comes from the data
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

#Model. The y is the y that we will be predeicting
y = tf.matmul(x,W) + b

#Loss function
#The actual labels are y_ and logits y is what our model is predicting
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#Train the model----------------------------
#For this example, we will use steepest gradient descent, with a step length of 0.5, to descend the cross entropy.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
	batch = mnist.train.next_batch(100)
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#Evaluate the model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Evaluating on testing data
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
