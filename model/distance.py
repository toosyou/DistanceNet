import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

class MultiHeadDistanceLayer(tf.keras.layers.Layer):
    def __init__(self, num_head, head_dim, max_length, global_mode, window_size=3, **kwargs):
        self.num_head = num_head
        self.head_dim = head_dim
        self.max_length = max_length
        self.global_mode = global_mode
        self.window_size = window_size
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

        # value
        self.value_embedding_weight = self.add_weight(shape=(input_dim, self.num_head),
                                                        initializer='GlorotNormal',
                                                        trainable=True, name='value_embedding_weight')

        self.learned_pe = self.add_weight(shape=(self.max_length, input_dim),
                                            initializer='GlorotNormal',
                                            trainable=True, name='learned_pe')

        # distance matrix of shape (max_length, max_length)
        self.distances_matrix = K.arange(self.max_length, dtype='float32')[None, :] - K.arange(self.max_length, dtype='float32')[:, None] 
        self.distances_matrix /= self.max_length

    def call(self, inputs, return_attention=False):
        query, key, value = inputs + self.learned_pe, inputs + self.learned_pe, inputs

        # get sizes
        data_length = query.shape[1]
        
        # embedding
        query = tf.matmul(query, self.query_embedding_weight)   # (?, data_length, num_head * head_dim)
        key   = tf.matmul(key, self.key_embedding_weight)       # (?, data_length, num_head * head_dim)
        value = tf.matmul(value, self.value_embedding_weight)   # (?, data_length, num_head)

        multi_head_query    = tf.concat(tf.split(query[None, ...], self.num_head, axis=3), axis=0) + self.query_embedding_bias      # (num_head, ?, data_length, head_dim)
        multi_head_key      = tf.concat(tf.split(key[None, ...], self.num_head, axis=3), axis=0)   + self.key_embedding_bias        # (num_head, ?, data_length, head_dim)
        multi_head_value    = K.permute_dimensions(value, (2, 0, 1))                                                                # (num_head, ?, data_length)
        
        # calculate distance attention
        attention = tf.matmul(multi_head_query, multi_head_key, transpose_b=True) * (float(self.head_dim) ** -0.5) # (num_head, ?, data_length, data_length)
        attention = tf.keras.layers.Softmax()(attention)
        attention = tf.linalg.band_part(attention, -1, 0) # lower triangle
        attention = attention * multi_head_value[..., None]

        # smoothen
        attention = K.pool2d(attention, (1, self.window_size), (1, 1), 'same', 'channels_first', 'avg')
        attention = K.permute_dimensions(attention, (1, 2, 3, 0)) # (?, data_length, data_length, num_head)

        if self.global_mode:
            output = K.sum(attention, axis=1)
        else:
            output = attention
        
        if return_attention:
            return output, attention
        return output

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], self.num_head]