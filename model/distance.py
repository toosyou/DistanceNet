import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

class MultiHeadDistanceLayer(tf.keras.layers.Layer):
    def __init__(self, num_head, head_dim, mode, output_dim=None, window_size=3, **kwargs):
        self.num_head = num_head
        self.head_dim = head_dim
        self.mode = mode
        self.window_size = window_size
        self.output_dim = output_dim

        assert self.mode in ('global', 'local'), 'mode must be either global or local'
        super().__init__(**kwargs)

    def get_config(self):
        config = {
            'num_head': self.num_head,
            'head_dim': self.head_dim,
            'mode': self.mode,
            'output_dim': self.output_dim,
            'window_size': self.window_size,
        }
        base_config = super().get_config()
        config.update(base_config)
        return config

    def build(self, input_shape):
        data_length, input_dim = input_shape[-2:] # (n_batch, data_length, feature_dim)

        self.query_embedding    = tf.keras.layers.Dense(self.num_head * self.head_dim, use_bias=True)
        self.key_embedding      = tf.keras.layers.Dense(self.num_head * self.head_dim, use_bias=True)
        self.value_embedding    = tf.keras.layers.Dense(self.num_head, use_bias=False)

        self.learned_pe = self.add_weight(shape=(data_length, input_dim),
                                            initializer='GlorotNormal',
                                            trainable=True, name='learned_pe')

        if self.mode == 'local': # local mode
            self.output_embedding = tf.keras.layers.Dense(self.output_dim, use_bias=False)

    def call(self, inputs, return_attention=False):
        query, key, value = inputs + self.learned_pe, inputs + self.learned_pe, inputs

        # get sizes
        data_length = query.shape[1]
        
        # embedding
        query   = self.query_embedding(query)               # (?, data_length, num_head * head_dim)
        key     = self.key_embedding(key)                   # (?, data_length, num_head * head_dim)
        value   = K.sigmoid(self.value_embedding(value))    # (?, data_length, num_head)

        multi_head_query    = tf.concat(tf.split(query[None, ...], self.num_head, axis=3), axis=0)      # (num_head, ?, data_length, head_dim)
        multi_head_key      = tf.concat(tf.split(key[None, ...], self.num_head, axis=3), axis=0)        # (num_head, ?, data_length, head_dim)
        multi_head_value    = K.permute_dimensions(value, (2, 0, 1))                                    # (num_head, ?, data_length)
        
        # calculate distance attention
        attention = tf.matmul(multi_head_query, multi_head_key, transpose_b=True) * (float(self.head_dim) ** -0.5) # (num_head, ?, data_length, data_length)
        attention = tf.keras.layers.Softmax()(attention)

        attention = tf.linalg.band_part(attention, 0, -1) # upper triangle
        # distance padding
        attention = tf.linalg.diag_part(attention, k=(0, data_length-1))
        attention = K.permute_dimensions(attention, (0, 1, 3, 2)) # transpose
        attention = K.reverse(attention, axes=(-2, -1))

        attention = attention * multi_head_value[..., None]

        # smoothen
        attention = K.pool2d(attention, (1, self.window_size), (1, 1), 'same', 'channels_first', 'avg')
        
        if self.mode == 'global':
            output = K.sum(attention, axis=2)                   # (num_head, ?, data_length)
            output = K.permute_dimensions(output, (1, 2, 0))    # (?, data_length, num_head)
        else:
            # do output embedding
            output = self.output_embedding(attention)                                       # (num_head, ?, data_length, output_dim)
            output = K.permute_dimensions(output, (1, 2, 3, 0))                             # (?, data_length, output_dim, num_head)
            output = K.reshape(output, (-1, data_length, self.output_dim * self.num_head))  # (?, data_length, output_dim * num_head)

        if return_attention:
            return output, K.permute_dimensions(attention, (1, 2, 3, 0)) # (?, data_length, data_length, num_head)
        return output

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], self.num_head]