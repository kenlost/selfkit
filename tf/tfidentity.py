import tensorflow as tf

x = tf.Variable(0, dtype=tf.int32, name='x')
old_val = tf.identity(x, name="old_same_x")
old_val = old_val + 10
new_val = tf.assign(x, x + 1, name="new_value")

writer = tf.summary.FileWriter('./tfid', tf.get_default_graph())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3):
        print(sess.run([new_val, old_val,  x]))