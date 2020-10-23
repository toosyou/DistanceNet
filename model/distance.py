import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

class MultiHeadDistanceLayer(tf.keras.layers.Layer):
    def __init__(self, num_head, head_dim, max_length, **kwargs):
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
        self.query_embedding_bias = self.add_weight(shape=(self.num_head, 1, 1, self.head_dim),
                                                        initializer='Zeros',
                                                        trainable=True, name='query_embedding_bias')

        # key
        self.key_embedding_weight = self.add_weight(shape=(input_dim, self.num_head * self.head_dim),
                                                        initializer='GlorotNormal',
                                                        trainable=True, name='key_embedding_weight')
        self.key_embedding_bias = self.add_weight(shape=(self.num_head, 1, 1, self.head_dim),
                                                        initializer='Zeros',
                                                        trainable=True, name='key_embedding_bias')

        # prior
        self.prior_mean = self.add_weight(shape=(self.num_head, ),
                                            initializer=tf.keras.initializers.RandomUniform(-1, 1),
                                            trainable=True, name='prior_mean')
        self.log_prior_std = self.add_weight(shape=(self.num_head, ),
                                            initializer=tf.keras.initializers.Constant(np.log(0.4)),
                                            trainable=True, name='log_prior_std')

        # distance matrix of shape (max_length, max_length)
        self.distances_matrix = K.arange(self.max_length, dtype='float32')[None, :] - K.arange(self.max_length, dtype='float32')[:, None] 
        self.distances_matrix /= self.max_length

    def call(self, inputs, return_attention=False):
        query, key = inputs, inputs

        # get sizes
        data_length = query.shape[1]
        
        # embedding
        query = tf.matmul(query, self.query_embedding_weight)   # (?, data_length, num_head * head_dim)
        key = tf.matmul(key, self.key_embedding_weight)         # (?, data_length, num_head * head_dim)

        multi_head_query    = tf.concat(tf.split(query[None, ...], self.num_head, axis=3), axis=0) + self.query_embedding_bias      # (num_head, ?, data_length, head_dim)
        multi_head_key      = tf.concat(tf.split(key[None, ...], self.num_head, axis=3), axis=0)   + self.key_embedding_bias        # (num_head, ?, data_length, head_dim)

        # prior
        prior_array = tf.repeat(self.distances_matrix[None, ...], self.num_head, axis=0)                                    # (num_head, data_length, data_length)
        prior_array = self.gaussian(prior_array, self.prior_mean[:, None, None], K.exp(self.log_prior_std[:, None, None]))  # (num_head, data_length, data_length)
        
        # calculate distance attention
        attention = tf.matmul(multi_head_query, multi_head_key, transpose_b=True) * (float(self.head_dim) ** -0.5) # (num_head, ?, data_length, data_length)
        attention = attention * prior_array[:, None, :data_length, :data_length]
        attention = tf.keras.layers.Softmax()(attention)
        
        # distance transform
        distance = attention * self.distances_matrix[:data_length, :data_length]    # (num_head, ?, data_length, data_length)
        distance = K.sum(distance, axis=-1)                                         # (num_head, ?, data_length)

        distance = K.permute_dimensions(distance, (1, 2, 0))                        # (?, data_length, num_head)

        if return_attention:
            return distance, K.permute_dimensions(attention, (1, 0, 2, 3))
        return distance

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], self.num_head]