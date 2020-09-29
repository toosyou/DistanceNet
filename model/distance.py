import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

class DistanceLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, model_dim, max_length, **kwargs):
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.max_length = max_length
        super().__init__(**kwargs)

    @staticmethod
    def gaussian(x, mean, std):
        return (1.0 / std / tf.sqrt(2.0 * 3.1415926))*tf.exp(-0.5 * (x - mean)**2.0 / std**2.0)

    def build(self, input_shape):
        # query
        self.query_embedding_weight = self.add_weight(shape=(self.input_dim, self.model_dim),
                                                        initializer='GlorotNormal',
                                                        trainable=True)
        self.query_embedding_bias = self.add_weight(shape=(self.model_dim, ),
                                                        initializer='Zeros',
                                                        trainable=True)

        # key
        self.key_embedding_weight = self.add_weight(shape=(self.input_dim, self.model_dim),
                                                        initializer='GlorotNormal',
                                                        trainable=True)
        self.key_embedding_bias = self.add_weight(shape=(self.model_dim, ),
                                                        initializer='Zeros',
                                                        trainable=True)

        # prior
        self.prior_mean = self.add_weight(shape=(1, ),
                                            initializer='Zeros',
                                            trainable=True)
        self.prior_std = self.add_weight(shape=(1, ),
                                            initializer='Ones',
                                            trainable=True)

        self.distances_matrix = K.arange(self.max_length, dtype='float32')[None, :] - K.arange(self.max_length, dtype='float32')[:, None] # i - j matrix

    def call(self, inputs):
        query, key = inputs, inputs
        
        # embedding
        query = tf.matmul(query, self.query_embedding_weight) + self.query_embedding_bias
        key = tf.matmul(key, self.key_embedding_weight) + self.key_embedding_bias

        # prior
        prior_array = self.gaussian(self.distances_matrix, self.prior_mean, self.prior_std)[None, :, :]

        # calculate distance attension
        attension = tf.matmul(query, key, transpose_b=True) * (float(self.model_dim) ** -0.5)
        attension = attension * prior_array[:, :query.shape[1], :query.shape[1]]
        attension = tf.keras.layers.Softmax()(attension)
        
        # distance transform
        distance = tf.matmul(attension, self.distances_matrix[:attension.shape[1], :attension.shape[1]])
        output = tf.linalg.diag_part(distance)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0:2]