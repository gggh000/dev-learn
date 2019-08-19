import tensorflow as tf
from numba import cuda
config=tf.ConfigProto()
config.log_device_placement = True
sess=tf.Session(config=config)

with tf.device("/cpu:0"):
        a = tf.Variable(3.0)
        b = tf.constant(4.0)

print(a)
print(b)

c=a*b

print(c)

#sess.run(c)





