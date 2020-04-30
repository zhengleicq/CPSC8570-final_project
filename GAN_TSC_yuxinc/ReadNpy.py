import tensorflow as tf
import numpy as np

#加载npy文件测试
w=8
weights_dict = np.load(f'teacher_models/mnist/mnist_adv_pretrain_{w}.npy',allow_pickle=True).item() #读入npy文件
#将字典中的某个值以张量的形式赋给网络中的某个权重和偏置（得知道键）
#trainable决定你是否要固定权重，False代表固定权重
b = tf.Variable(data['weight:0'], dtype=tf.float32, trainable=False)
sess=tf.Session()  
sess.run(tf.global_variables_initializer())
print(sess.run(b))


for op_name in weights_dict:
    print(op_name)
    for data in weights_dict[op_name]:
        if len(data.shape) == 1:
            var = tf.get_variable('biases', trainable=False)
            session.run(var.assign(data))
        else:
            var = tf.get_variable('weights', trainable=False)
            session.run(var.assign(data))