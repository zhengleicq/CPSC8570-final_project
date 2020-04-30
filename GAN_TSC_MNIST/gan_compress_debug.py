"""
Code modified from: https://github.com/chengshengchan/model_compression
"""
import tensorflow as tf
import numpy as np
import os, sys
import argparse
from model import teacher_lenet, student_lenet

def parse_args():
    parser = argparse.ArgumentParser(description='teacher-student model')    
    parser.add_argument('--p_fake', dest='p_fake', default=0, help='probability of training on GAN data', type=float)
    parser.add_argument('--model_path', dest='model_path', default='generate_models/mnist/ACGAN_mnist_keras15.h5', help="path to trained GAN model.", type=str)#generate model
    parser.add_argument('--teacher_w', dest='teacher_w', default=8, help="teacher model scale factor", type=int)#generate model
    parser.add_argument('--student_w', dest='student_w', default=4, help="student model scale factor", type=int)#generate model
    #parser.add_argument('--teacher_path', dest='teacher_path', default='teacher_models/mnist/mnist_adv_pretrain_8.npy', help="path to trained GAN model.", type=str)#generate model
    parser.add_argument('--student_path', dest='student_path', default='student_models/mnist/mnist_adv_gan_4.npy', help="path to save trained GAN model.", type=str)#generate model
    parser.add_argument('--lr', dest='lr', default=1e-4, help='learning rate', type=float)
    parser.add_argument('--epoch', dest='epoch', default=5, help='total epoch', type=int)
    parser.add_argument('--batch_size', dest='batch_size', default=64, help="batch size", type=int)
    parser.add_argument('--gpu', dest='gpu', default=0, help="which gpu to use", type=int)    
    args = parser.parse_args()
    return args, parser

def perturb(x_nat, sess, x, grad, keep_prob, dropout_rate):
# def perturb(x_nat, sess, x, grad, keep_prob, dropout_rate):
    #print('#############pertubation################')
    epsilon=0.3
    a=0.01
  
    if True:
        x_ = x_nat + np.random.uniform(-epsilon, epsilon, x_nat.shape)
        x_ = np.clip(x_, 0, 1) # ensure valid pixel range
    else:
        x_ = np.copy(x_nat)
  
    #   grad = tf.gradients(loss, x)[0]
    for i in range(40):
        grad_ = sess.run(grad, feed_dict={x: x_nat,
                                          keep_prob: 1 - dropout_rate})
    x_ += a * np.sign(grad_)
    x_ = np.clip(x_, x_nat - epsilon, x_nat + epsilon) 
    x_ = np.clip(x_, 0, 1) # ensure valid pixel range

    return x_

def main():
    # Parameters
    lr = args.lr
    model_path = args.model_path
    teacher_w = args.teacher_w
    student_w = args.student_w
    teacher_path = f"./teacher_models/mnist/mnist_adv_pretrain_{teacher_w}.npy"
    student_load_path = f"./teacher_models/mnist/mnist_adv_pretrain_{student_w}.npy"
    student_save_path = f"./student_models/mnist/mnist_adv_gan_{student_w}.npy"
    total_epoch = args.epoch
    batch_size = args.batch_size
    p_fake = args.p_fake
    
    tf.reset_default_graph()
    # Placeholders
    x = tf.placeholder(tf.float32, [batch_size, dim, dim, 1])
    
    # Load Data
    (data, label), (data_test, label_test) = tf.keras.datasets.mnist.load_data()
    mean = np.mean(data, axis=0)
    index = np.array(range(len(data))) 
    iterations = int(len(data)/batch_size)    
    
    # Load Model and Basic Settings
    #teacher=lenet_teacher(x, keep_prob,teacher_w)
    teacher, t_weights_list = teacher_lenet(x,teacher_w)
    print("teacher weights list: ", t_weights_list)
    student, s_weights_list = student_lenet(x,student_w)
    print("student weights list: ", s_weights_list)
    #student=student_lenet(x, w=4)
    # Generator Network Variables
    #tea_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='teacher')
    # Discriminator Network Variables
    stu_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='student')
    print("student vars: ", stu_vars)
    tf_loss = tf.nn.l2_loss(teacher - student)/batch_size
    grad = tf.gradients(tf_loss, x)[0]
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(tf_loss, var_list=stu_vars)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)
    tf.global_variables_initializer().run()
    with tf.device('/cpu:0'):
        saver = tf.train.Saver(max_to_keep=100)

    # Train
    print('Start Training')
    for i in range(total_epoch):
        np.random.shuffle(index)
        cost_sum = 0
        total = 0
        
        teacher_weight = np.load(teacher_path,allow_pickle=True).item()
        t_weights_list[0].load(teacher_weight['layer_conv1/kernel'],sess)
        t_weights_list[1].load(teacher_weight['layer_conv2/kernel'],sess)
        t_weights_list[2].load(teacher_weight['layer_fc1/kernel'],sess)
        t_weights_list[3].load(teacher_weight['layer_fc_out/kernel'],sess)
        
        student_weight = np.load(student_load_path,allow_pickle=True).item()
        s_weights_list[0].load(student_weight['layer_conv1/kernel'],sess)
        s_weights_list[1].load(student_weight['layer_conv2/kernel'],sess)
        s_weights_list[2].load(student_weight['layer_fc1/kernel'],sess)
        s_weights_list[3].load(student_weight['layer_fc_out/kernel'],sess)
        # Generate GAN data
        if p_fake > 0:
            data_acgan = generate_fake(int(p_fake*len(data)), model_path)
            j_acgan = 0
            index_acgan = np.array(range(len(data_acgan))) 
            np.random.shuffle(index_acgan)           
                  
        for j in range(iterations):
            if np.random.rand() > p_fake: # Train on real training data 
                batch_x = data[index[j*batch_size:(j+1)*batch_size]]
            else: # Train on GAN data
                if (j_acgan+1)*batch_size < len(data_acgan):
                    batch_x = data_acgan[index_acgan[j_acgan*batch_size:(j_acgan+1)*batch_size]]
                    j_acgan += 1
                else:
                    j_rand = np.random.randint(j_acgan)
                    batch_x = data_acgan[index_acgan[j_rand*batch_size:(j_rand+1)*batch_size]]  
            batch_x = np.reshape(batch_x, (batch_size, dim, dim))
            batch_x = np.float32(batch_x) - mean
            batch_x = np.reshape(batch_x, (batch_size, dim, dim, 1))
            batch_x_p = perturb(batch_x, sess, x, grad, keep_prob, dropout_rate)
            _, cost = sess.run([optimizer, tf_loss],
                                feed_dict={x : batch_x_p, keep_prob : 1-dropout_rate}) 
            #_, cost = sess.run([optimizer, tf_loss],
            #                   feed_dict={x : batch_x})      
            total += batch_size                     
            cost_sum += cost
        print ("Epoch %d || Training cost = %.2f"%(i, cost_sum/iterations/n_classes))

        
    # Test
    pred = tf.nn.softmax(student)
    total = 0
    correct = 0
    cost_test = 0
    iterations_test = int(len(data_test)/batch_size)
    for j in range(iterations_test):
        batch_x = data_test[j*batch_size:(j+1)*batch_size] - mean
        batch_x = np.reshape(batch_x, (batch_size, dim, dim,1))
        batch_x_p = perturb(batch_x, sess, x, grad, keep_prob, dropout_rate)
        prob_p, cost_p = sess.run([pred, tf_loss],
                feed_dict={x : batch_x_p, keep_prob : 1.0})
        #prob, cost = sess.run([pred, tf_loss],
                #feed_dict={x : batch_x})
        label_batch = label_test[j*batch_size:(j+1)*batch_size].reshape(-1)
        #pred_batch = np.array( [np.argmax(prob[i]) for i in range(prob.shape[0])])
        pred_batch_p = np.array( [np.argmax(prob_p[i]) for i in range(prob_p.shape[0])])
        correct_p += sum(label_batch == pred_batch_p)
        #correct += sum(label_batch == pred_batch)
        total += batch_size
        #cost_test += cost
        cost_test_p += cost_p
    print ("\nEnd of Training\nTest nat acc = %.4f || Test nat cost = %.2f\n"%(float(correct)/total, cost_test/iterations_test/n_classes))
    print ("\nEnd of Training\nTest adv acc = %.4f || Test adv cost = %.2f\n"%(float(correct_p)/total_p, cost_test_p/iterations_test/n_classes))

if __name__ == '__main__':
    args, parser = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Parameters for CIFAR-10
    dim = 28
    n_classes = 10
    
    # Functions to read Keras or Pytorch Models
    if 'keras' in args.model_path:
        from functions.mnist.generate_fake_keras import generate_fake
    #elif 'pytorch' in args.model_path:
        #from functions.mnist.generate_fake_pytorch import generate_fake
    else:
        sys.exit('ERROR: model_path is not valid. Default must include - keras / pytorch - in the model name. New model please refer to the code and make corresponding modification.')
    
    main()