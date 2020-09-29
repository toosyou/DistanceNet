import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

class MultiHeadDistanceLayer(tf.keras.layers.Layer):
    def __init__(self, num_head, head_dim, max_length, **kwargs):
        self.num_head = num_head
        self.head_dim = head_dim
        self.max_length = max_length
        super().__init__(**kwargs)

    @staticmethod
    def gaussian(x, mean, std):
        return (1.0 / std / tf.sqrt(2.0 * 3.1415926))*tf.exp(-0.5 * (x - mean)**2.0 / std**2.0)

    def build(self, input_shape):
        input_dim = input_shape[-1] # (n_batch, data_length, feature_dim)

        # query
        self.query_embedding_weight = self.add_weight(shape=(input_dim, self.num_head * self.head_dim),
                                                        initializer='GlorotNormal',
                                                        trainable=True)
        self.query_embedding_bias = self.add_weight(shape=(self.num_head, ),
                                                        initializer='Zeros',
                                                        trainable=True)

        # key
        
        self.key_embedding_weight = self.add_weight(shape=(input_dim, self.num_head * self.head_dim),
                                                        initializer='GlorotNormal',
                                                        trainable=True)
        self.key_embedding_bias = self.add_weight(shape=(self.num_head, ),
                                                        initializer='Zeros',
                                                        trainable=True)

        # prior
        self.prior_mean = self.add_weight(shape=(1, ),
                                            initializer='Zeros',
                                            trainable=True)
        self.prior_std = self.add_weight(shape=(1, ),
                                            initializer='Ones',
                                            trainable=True)

        # distance matrix of shape (max_length, max_length)
        self.distances_matrix = K.arange(self.max_length, dtype='float32')[None, :] - K.arange(self.max_length, dtype='float32')[:, None] 

    def call(self, inputs):
        query, key = inputs, inputs

        # get sizes
        data_length = query.shape[1]
        
        # embedding
        query = tf.matmul(query, self.query_embedding_weight) + tf.repeat(self.query_embedding_bias, self.head_dim) # (?, data_length, num_head * head_dim)
        key = tf.matmul(key, self.key_embedding_weight) + tf.repeat(self.key_embedding_bias, self.head_dim)         # (?, data_length, num_head * head_dim)

        multi_head_query    = tf.concat(tf.split(query, self.num_head, axis=2), axis=0)     # (num_head * ?, data_length, head_dim)
        multi_head_key      = tf.concat(tf.split(key, self.num_head, axis=2), axis=0)       # (num_head * ?, data_length, head_dim)
        
        # prior
        prior_array = self.gaussian(self.distances_matrix, self.prior_mean, self.prior_std) # (data_length, data_length)

        # calculate distance attension
        attension = tf.matmul(multi_head_query, multi_head_key, transpose_b=True) * (float(self.head_dim) ** -0.5) # (num_head * ?, data_length, data_length)
        attension = attension * prior_array[:data_length, :data_length]
        attension = tf.keras.layers.Softmax()(attension)
        
        # distance transform
        distance = attension * self.distances_matrix[:data_length, :data_length]
        distance = K.sum(distance, axis=-1)

        distance = K.expand_dims(distance, axis=-1)
        distance = tf.concat(tf.split(distance, self.num_head, axis=0), axis=-1) # (?, data_length, num_head)

        return distance

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], self.num_head]