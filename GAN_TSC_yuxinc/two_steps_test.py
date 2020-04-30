
# coding: utf-8

# get_ipython().run_line_magic('matplotlib', 'inline')
import sys
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
tf.logging.set_verbosity(tf.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))



data.test.cls = np.argmax(data.test.labels, axis=1)


# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

#w=8
def model_2(x_image,w):

    conv1 = tf.layers.conv2d(inputs=x_image, name='layer_conv1', padding='same',
                           filters=2*w, kernel_size=5, activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, name='layer_conv2', padding='same',
                           filters=4*w, kernel_size=5, activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)

    flatten = tf.layers.flatten(pool2)

    fc1 = tf.layers.dense(inputs=flatten, name='layer_fc1',
                          units=64*w, activation=tf.nn.relu)

    logits = tf.layers.dense(inputs=fc1, name='layer_fc_out',
                          units=num_classes, activation=None)

    return logits


def perturb(x_nat, y, sess, x, y_true, grad):
    """Given a set of examples (x_nat, y), returns a set of adversarial
     examples within epsilon of x_nat in l_infinity norm."""
  
    epsilon=0.3
    a=0.01
    
    if True:
        x_ = x_nat + np.random.uniform(-epsilon, epsilon, x_nat.shape)
        x_ = np.clip(x_, 0, 1) # ensure valid pixel range
    else:
        x_ = np.copy(x_nat)
  
    #   grad = tf.gradients(loss, x)[0]
    for i in range(40):
        grad_ = sess.run(grad, feed_dict={x: x_,
                                          y_true: y})

        x_ += a * np.sign(grad_)
        x_ = np.clip(x_, x_nat - epsilon, x_nat + epsilon) 
        x_ = np.clip(x_, 0, 1) # ensure valid pixel range

    return x_



def train(model,w):
    print("training model")
    tf.reset_default_graph()
    
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)
    
    train_loss = []
    train_accuracy = []
#     init, optimizer, loss, accuracy = model(x, x_image, y_true, y_true_cls)
    logits = model(x_image,w)
    
    y_pred = tf.nn.softmax(logits=logits)
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    opt = tf.train.AdamOptimizer(learning_rate=1e-4)
    optimizer = opt.minimize(loss)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    grad = tf.gradients(loss, x)[0]
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    
    with tf.Session() as session:

        train_batch_size = 64

        session.run(init)

        for i in range(0, 1000):
        
#             saver.restore(session,tf.train.latest_checkpoint('./checkpoint_dir_adv_w_8'))

            x_batch, y_true_batch = data.train.next_batch(train_batch_size)
            
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}
            
            x_perturb = perturb(x_batch, y_true_batch, session, x, y_true, grad)
#             for j in range(300, 400):
#                 if x_batch[14][j]!=0:
#                     print('x_batch', x_batch[14][j])
#                     print('x_perturb', x_perturb[14][j])
#                     print(x_perturb.shape)

#             session.run(optimizer, feed_dict={x:x_perturb, y_true:y_true_batch})
            session.run(optimizer, feed_dict=feed_dict_train)
#             los, acc = session.run([loss, accuracy], feed_dict=feed_dict_train)
            
#             los_perturb, acc_perturb = session.run([loss, accuracy], feed_dict={x:x_perturb, y_true:y_true_batch})
             
            if i % 10 == 0:
                if i%1000==0:
                    saver.save(session, './checkpoint_dir_adv_w_8/MyModel')
                    
                x_test_batch, y_test_batch = data.test.next_batch(1000)
                x_test_perturb = perturb(x_test_batch, y_test_batch, session, x, y_true, grad)
                nat_acc = session.run(accuracy,feed_dict={x:x_test_batch, y_true:y_test_batch})
                adv_acc = session.run(accuracy,feed_dict={x:x_test_perturb, y_true:y_test_batch})
            
                print("nat_acc = ",nat_acc)
                print("adv_acc = ", adv_acc)
#     return train_loss, train_accuracy
    tf.get_default_graph().finalize() 


def retrain(model,w):
    tf.reset_default_graph()
    
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)
    
    train_loss = []
    train_accuracy = []
#     init, optimizer, loss, accuracy = model(x, x_image, y_true, y_true_cls)
    logits = model(x_image,w)
    
    y_pred = tf.nn.softmax(logits=logits)
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    opt = tf.train.AdamOptimizer(learning_rate=1e-4)
    optimizer = opt.minimize(loss)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     init = tf.global_variables_initializer()
    grad = tf.gradients(loss, x)[0]
    saver = tf.train.Saver()
    
    with tf.Session() as session:
        print(f'read from ./checkpoint_dir_adv_w_{w}')
        saver.restore(session,tf.train.latest_checkpoint(f'./checkpoint_dir_adv_w_{w}'))

        train_batch_size = 64

#         session.run(init)

        for i in range(0, 168000):

            x_batch, y_true_batch = data.train.next_batch(train_batch_size)
            
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}
            
            x_perturb = perturb(x_batch, y_true_batch, session, x, y_true, grad)
#             for j in range(300, 400):
#                 if x_batch[14][j]!=0:
#                     print('x_batch', x_batch[14][j])
#                     print('x_perturb', x_perturb[14][j])
#                     print(x_perturb.shape)

            session.run(optimizer, feed_dict={x:x_perturb, y_true:y_true_batch})
#             session.run(optimizer, feed_dict=feed_dict_train)
#             los, acc = session.run([loss, accuracy], feed_dict=feed_dict_train)
            
#             los_perturb, acc_perturb = session.run([loss, accuracy], feed_dict={x:x_perturb, y_true:y_true_batch})
             
            if i % 100 == 0:
                if i%1000 == 0:
                    print(f'save to ./test_model/checkpoint_dir_adv_w_{w}/MyModel')
                    saver.save(session, f'./test_model/checkpoint_dir_adv_w_{w}/MyModel')
                
                x_test_batch, y_test_batch = data.test.next_batch(10000)
                x_test_perturb = perturb(x_test_batch, y_test_batch, session, x, y_true, grad)
                nat_acc = session.run(accuracy,feed_dict={x:x_test_batch, y_true:y_test_batch})
                adv_acc = session.run(accuracy,feed_dict={x:x_test_perturb, y_true:y_test_batch})
                print('step = ', i)
                print("nat_acc = ",nat_acc)
                print("adv_acc = ", adv_acc)
#     return train_loss, train_accuracy
    tf.get_default_graph().finalize() 

def main():
    w = int(sys.argv[1])
    #train(model_2,w)
    print(w)
    retrain(model_2,w)


if __name__ == '__main__':
    main()
