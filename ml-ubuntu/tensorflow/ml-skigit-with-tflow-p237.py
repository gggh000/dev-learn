import tensorflow as tf
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
        print(y.eval())
        print(z.eval())







