import tensorflow as tf
t=tf.constant([[1.,2.,3.],[4.,5.,6.]]) 
print(t)
print(t.shape)
print(t.dtype)
print(t[:,1:])
print(t[..., 1, tf.newaxis])
print(t+10)
print(tf.square(t))
print(tf.transpose(t))

