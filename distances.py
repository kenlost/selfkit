
"""
Compute distance from an arrays
"""

import tensorflow as tf
import sqlite3
import numpy as np
import time

def get_face_v(database_filename='face.db'):

    scope_conn = sqlite3.connect(database_filename)
    scope_cursor = scope_conn.cursor()
    records = []
    sql_statement = 'SELECT vector ' + ' FROM image_face_table'

    scope_cursor.execute(sql_statement)
    records = scope_cursor.fetchall()
    return records

def bad_culute():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        with tf.Session() as sess:
            data = tf.placeholder(tf.float32, (1248, 128), 'input')
            face1 = tf.placeholder(tf.float32, (128,), 'face')
            # tf.reduce_sum(data, data)
            # data_t = tf.transpose(data,[1,0])
            # dis = tf.sqrt(tf.reduce_sum(tf.square(data[i] - data[j]), axis=1))
            _distance = tf.sqrt(tf.reduce_sum(tf.square(data - face1), 1))

            begin = time.time()
            face_v = get_face_v()
            facev = []
            for face in face_v:
                # print(face)
                str_list = face[0].split(',')
                vector_data = list(map(lambda s: float(s), str_list))
                facev.append(vector_data)
            facev = np.asarray(facev)

            print(facev.shape)
            time_load = time.time()
            print("load vector %.2f " % (time_load - begin))
            init = tf.global_variables_initializer()
            distances = []
            for face in facev:
                distance = sess.run(_distance, feed_dict={data: facev, face1: face})
                distances.append(distance)

            time_exe = time.time()
            print("cult distance vector %.2f " % (time_exe - begin))
            print(len(distances))
            print(distances)


def cutlt_distance_GPU():
    with tf.Graph().as_default(), tf.device('/gpu:0'):
        with tf.Session() as sess:
            # len = 1248
            data = tf.placeholder(tf.float32, (None, 128), 'input')
            x1 = tf.expand_dims(data, 0)
            print(x1.shape)
            len = tf.shape(x1)
            print(len)
            xo = tf.tile(x1, [len[1], 1, 1])
            xt = tf.transpose(xo, [1, 0, 2])

            _distance = tf.sqrt(tf.reduce_sum(tf.square(xo-xt), 2))

            begin = time.time()
            face_v = get_face_v()
            facev = []
            for face in face_v:
                # print(face)
                str_list = face[0].split(',')
                vector_data = list(map(lambda s: float(s), str_list))
                facev.append(vector_data)
            facev = np.asarray(facev)

            print(facev.shape)
            time_load = time.time()
            print("load vector %.2f " % (time_load - begin))
            init = tf.global_variables_initializer()

            distances = sess.run(_distance, feed_dict={data: facev })

            time_exe = time.time()
            print("cult distance vector %.2f " % (time_exe - begin))
            print(distances.shape)
            print(distances)

def dist(face1, face2):
    distance = np.sqrt(np.sum(np.square(face1 - face2)))
    # print(distance)
    return distance

def cult_with_cpu():
    begin = time.time()
    face_v = get_face_v()
    facev = []
    for face in face_v:
        # print(face)
        str_list = face[0].split(',')
        vector_data = list(map(lambda s: float(s), str_list))
        facev.append(vector_data)

    facev = np.asarray(facev)

    print(facev.shape)
    time_load = time.time()
    print("load vector %.2f " % (time_load - begin))
    distances = np.zeros((len(facev),len(facev)), dtype=np.float32)
    print(distances.shape)
    for i,face in enumerate(facev):
        for k, face2 in enumerate(facev):
            # distances[i][k] = dist(face, face2)
            if k == i:
                pass
            elif k<i:
                distances[i][k] = distances[k][i]
            else:
                distances[i][k] = dist(face, face2)

    time_exe = time.time()
    print("cult distance vector %.2f " % (time_exe - begin))
    print(distances.shape)
    # print(distances)

def cult_with_np():
    begin = time.time()
    face_v = get_face_v()
    facev = []
    for face in face_v:
        # print(face)
        str_list = face[0].split(',')
        vector_data = list(map(lambda s: float(s), str_list))
        facev.append(vector_data)

    facev = np.asarray(facev)

    print(facev.shape)
    time_load = time.time()
    print("load vector %.2f " % (time_load - begin))
    distances = np.zeros((len(facev),len(facev)), dtype=np.float32)
    print(distances.shape)
    for i,face in enumerate(facev):
        distance_l = np.linalg.norm(face - facev, axis=1)
        distances[i,:] = distance_l
        # for k, face2 in enumerate(facev):
        #     # distances[i][k] = dist(face, face2)
        #     if k == i:
        #         pass
        #     elif k<i:
        #         distances[i][k] = distances[k][i]
        #     else:
        #         distances[i][k] = dist(face, face2)

    time_exe = time.time()
    print("cult distance vector %.2f " % (time_exe - begin))
    print(distances.shape)
    print(distances)


if __name__ == '__main__':
    # import timeit
    # cult_with_cpu()
    cult_with_np()
    cutlt_distance_GPU()
