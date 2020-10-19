import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

def get_positional_encoding(input_shape, temperature=10000):
    batch_size, signal_length, feature_dim = input_shape

    embed = tf.range(signal_length, dtype=tf.float32)# (signal_length)
    
    dim_t = tf.range(feature_dim, dtype=tf.float32) # (feature_dim)
    dim_t = temperature ** (2 * (dim_t // 2) / feature_dim)

    pos = embed[..., tf.newaxis] / dim_t # (signal_length, feature_dim)
    pos = tf.stack([tf.math.sin(pos[..., 0::2]), tf.math.cos(pos[..., 1::2])], axis=2)

    pos = tf.reshape(pos, [1, signal_length, -1])

    return pos # (1, signal_length, feature_dim)

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

        # value
        self.value_embedding_weight = self.add_weight(shape=(input_dim, self.num_head * self.head_dim),
                                                        initializer='GlorotNormal',
                                                        trainable=True, name='value_embedding_weight')
        self.value_embedding_bias = self.add_weight(shape=(self.num_head, ),
                                                        initializer='Zeros',
                                                        trainable=True, name='value_embedding_bias')

        # distance matrix of shape (max_length, max_length)
        self.distances_matrix = K.arange(self.max_length, dtype='float32')[None, :] - K.arange(self.max_length, dtype='float32')[:, None] 

        self.position_embedding = get_positional_encoding(input_shape)

    def call(self, inputs, return_attention=False):
        # query, key = inputs, inputs
        query = key = inputs + self.position_embedding

        value = inputs

        # get sizes
        data_length = query.shape[1]
        
        # embedding
        query = tf.matmul(query, self.query_embedding_weight) + tf.repeat(self.query_embedding_bias, self.head_dim) # (?, data_length, num_head * head_dim)
        key = tf.matmul(key, self.key_embedding_weight) + tf.repeat(self.key_embedding_bias, self.head_dim)         # (?, data_length, num_head * head_dim)
        value = tf.matmul(value, self.value_embedding_weight) + tf.repeat(self.value_embedding_bias, self.head_dim)         # (?, data_length, num_head * head_dim)

        multi_head_query    = tf.concat(tf.split(query, self.num_head, axis=2), axis=0)     # (num_head * ?, data_length, head_dim)
        multi_head_key      = tf.concat(tf.split(key, self.num_head, axis=2), axis=0)       # (num_head * ?, data_length, head_dim)
        multi_head_value      = tf.concat(tf.split(value, self.num_head, axis=2), axis=0)       # (num_head * ?, data_length, head_dim)
        
        # calculate distance attention
        attention = tf.matmul(multi_head_query, multi_head_key, transpose_b=True) * (float(self.head_dim) ** -0.5) # (num_head * ?, data_length, data_length)
        attention = tf.keras.layers.Softmax()(attention)

        distance = tf.matmul(attention, multi_head_value) # (num_head * ?, data_length, head_dim)

        # distance = K.expand_dims(distance, axis=-1)
        distance = tf.concat(tf.split(distance, self.num_head, axis=0), axis=-1) # (?, data_length, num_head)

        if return_attention:
            return distance, tf.concat(tf.split(attention[..., None], self.num_head, axis=0), axis=-1) # (?, data_length, num_head), (?, data_length, data_length, num_head)
        return distance

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], self.num_head]