import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
        print(y.eval())
        print(z.eval())







