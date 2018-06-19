import tensorflow as tf
import numpy as np
import os

import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()

# dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
base_dir = os.path.dirname(__file__)
# for one_element in tfe.Iterator(dataset):
#     print(one_element)
# 函数的功能时将filename对应的图片文件读进来，并缩放到统一的大小
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    print(tf.shape(image_decoded))
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_resized, label

# 图片文件的列表
filenames = tf.constant(["C:\\Users\\kenneth\\autoencoder\\image\\1.jpg",
                         "C:\\Users\\kenneth\\autoencoder\\image\\2.jpg",
                         "C:\\Users\\kenneth\\autoencoder\\image\\3.jpg",
                         "C:\\Users\\kenneth\\autoencoder\\image\\4.jpg",
                         "C:\\Users\\kenneth\\autoencoder\\image\\5.jpg",
                         "C:\\Users\\kenneth\\autoencoder\\image\\6.jpg"])
# label[i]就是图片filenames[i]的label
labels = tf.constant([0, 1, 2, 3, 4, 37])

# 此时dataset中的一个元素是(filename, label)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

# 此时dataset中的一个元素是(image_resized, label)
dataset = dataset.map(_parse_function)

# 此时dataset中的一个元素是(image_resized_batch, label_batch)
# dataset = dataset.shuffle(buffer_size=1000).batch(2).repeat(1)
dataset = dataset.batch(2).repeat(2)
# tfe.
# dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
#
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(5):
        print(sess.run(one_element)[1])