import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

class MultiHeadDistanceLayer(tf.keras.layers.Layer):
    def __init__(self, num_head, head_dim, max_length, prior_mean=0.0, prior_std=1.0, **kwargs):
        self.num_head = num_head
        self.head_dim = head_dim
        self.max_length = max_length
        super().__init__(**kwargs)

    def get_config(self):
        config = {
            'num_head': self.num_head,
            'head_dim': self.head_dim,
            'max_length': self.max_length
        }
        base_config = super().get_config()
        config.update(base_config)
        return config

    @staticmethod
    def gaussian(x, mean, std):
        return (1.0 / std / tf.sqrt(2.0 * 3.1415926))*tf.exp(-0.5 * (x - mean)**2.0 / std**2.0)

    def build(self, input_shape):
        input_dim = input_shape[-1] # (n_batch, data_length, feature_dim)

        # query
        self.query_embedding_weight = self.add_weight(shape=(input_dim, self.num_head * self.head_dim),
                                                        initializer='GlorotNormal',
                                                        trainable=True, name='query_embedding_weight')
        self.query_embedding_bias = self.add_weight(shape=(self.num_head, ),
                                                        initializer='Zeros',
                                                        trainable=True, name='query_embedding_bias')

        # key
        self.key_embedding_weight = self.add_weight(shape=(input_dim, self.num_head * self.head_dim),
                                                        initializer='GlorotNormal',
                                                        trainable=True, name='key_embedding_weight')
        self.key_embedding_bias = self.add_weight(shape=(self.num_head, ),
                                                        initializer='Zeros',
                                                        trainable=True, name='key_embedding_bias')

        # prior
        self.prior_mean = self.add_weight(shape=(self.num_head, ),
                                            initializer=tf.keras.initializers.RandomUniform(-self.max_length, self.max_length),
                                            trainable=True, name='prior_mean')
        self.log_prior_std = self.add_weight(shape=(self.num_head, ),
                                            initializer=tf.keras.initializers.Constant(np.log(self.max_length / 4.)),
                                            trainable=True, name='log_prior_std')

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
        prior_array = tf.repeat(self.distances_matrix[None, ...], self.num_head, axis=0)                                    # (num_head, data_length, data_length)
        prior_array = self.gaussian(prior_array, self.prior_mean[:, None, None], K.exp(self.log_prior_std[:, None, None]))  # (num_head, data_length, data_length)
        prior_array = tf.repeat(prior_array, tf.shape(inputs)[0], axis=0)                                                   # (num_head * ?, data_length, data_length)

        # calculate distance attension
        attension = tf.matmul(multi_head_query, multi_head_key, transpose_b=True) * (float(self.head_dim) ** -0.5) # (num_head * ?, data_length, data_length)
        attension = attension * prior_array[:, :data_length, :data_length]
        attension = tf.keras.layers.Softmax()(attension)
        
        # distance transform
        distance = attension * self.distances_matrix[:data_length, :data_length]
        distance = K.sum(distance, axis=-1)

        distance = K.expand_dims(distance, axis=-1)
        distance = tf.concat(tf.split(distance, self.num_head, axis=0), axis=-1) # (?, data_length, num_head)

        return distance

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], self.num_head]