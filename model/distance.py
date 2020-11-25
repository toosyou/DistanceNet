import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

class MultiHeadDistanceLayer(tf.keras.layers.Layer):
    def __init__(self, num_head, head_dim, mode, output_dim=None, window_size=3, distance_norm=False, max_distance=np.Inf, **kwargs):
        self.num_head = num_head
        self.head_dim = head_dim
        self.mode = mode
        self.output_dim = output_dim
        self.window_size = window_size
        self.distance_norm = distance_norm
        self.max_distance = max_distance

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

        self.max_distance = np.clip(self.max_distance, 0, data_length-1)

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

        # distance padding
        attention = tf.linalg.diag_part(attention, k=(-self.max_distance, self.max_distance)) # (num_head, ?, data_length, 2 * max_d + 1)
        attention = K.permute_dimensions(attention, (0, 1, 3, 2)) # transpose
        attention = K.reverse(attention, axes=(-2, -1))

        attention = attention * multi_head_value[..., None]

        # smoothen
        attention = K.pool2d(attention, (1, self.window_size), (1, 1), 'same', 'channels_first', 'avg')

        if self.distance_norm:
            attention = DistanceNorm()(attention)
        
        if self.mode == 'global':
            output = K.sum(attention, axis=2)                   # (num_head, ?, 2 * max_d + 1)
            output = K.permute_dimensions(output, (1, 2, 0))    # (?, 2 * max_d + 1, num_head)
        else:
            output = self.output_embedding(attention)                                       # (num_head, ?, data_length, output_dim)
            output = K.permute_dimensions(output, (1, 2, 3, 0))                             # (?, data_length, output_dim, num_head)
            output = K.reshape(output, (-1, data_length, self.output_dim * self.num_head))  # (?, data_length, output_dim * num_head)

        if return_attention:
            return output, K.permute_dimensions(attention, (1, 2, 3, 0)) # (?, data_length, data_length, num_head)
        return output

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], self.num_head]

class DistanceNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        max_distance = input_shape[-1]
        self.range_max_distance = tf.range(max_distance, dtype=tf.float32) - max_distance / 2. # (max_distance)

    @staticmethod
    def interpolated_gather_nd(source, indices):
        original_shape = tf.shape(source)
        data_length = original_shape[-2]
        max_distance = original_shape[-1]

        integer_indices = K.cast(indices, tf.int32)

        floor_indices = K.clip(integer_indices, 0, max_distance-1)
        ceil_indices = K.clip(integer_indices+1, 0, max_distance-1)

        floor_indices = K.expand_dims(K.expand_dims(floor_indices, axis=1), axis=-1)            # (-1, 1, max_distance, 1)
        ceil_indices = K.expand_dims(K.expand_dims(ceil_indices, axis=1), axis=-1)              # (-1, 1, max_distance, 1)

        floor_indices = tf.repeat(floor_indices, data_length, axis=1)                           # (-1, data_length, max_distance, 1)
        ceil_indices = tf.repeat(ceil_indices, data_length, axis=1)                             # (-1, data_length, max_distance, 1)

        weights = indices - tf.math.floor(indices)
        weights = K.expand_dims(weights, axis=1)
        weights = tf.repeat(weights, data_length, axis=1)
    
        # interpolation
        normed_distance_floor = tf.gather_nd(source, floor_indices, batch_dims=2)
        normed_distance_ceil = tf.gather_nd(source, ceil_indices, batch_dims=2)
        normed_distance = normed_distance_ceil * weights + normed_distance_floor * (1. - weights)

        return normed_distance

    def get_mean_std(self, distance):
        px    = K.sum(distance, axis=-2)  # (-1, max_distance)
        px    = px / K.sum(px, axis=-1)[:, None]   # (-1, max_distance), normed

        mean  = K.sum(px * self.range_max_distance, axis=-1) # (-1)
        std   = K.sqrt(K.sum(px * K.pow(self.range_max_distance[None, :] - mean[:, None], 2), axis=-1)) # (-1)

        return mean, std

    def call(self, distance):
        '''distance: (..., data_length, max_distance)
        '''
        original_shape = tf.shape(distance)
        data_length = original_shape[-2]
        max_distance = original_shape[-1]

        distance = K.reshape(distance, (-1, data_length, max_distance))

        mean, std = self.get_mean_std(distance) # (-1), (-1)
        new_indices = (self.range_max_distance[None, :] - mean[:, None]) / std[:, None] + K.cast(max_distance, dtype=tf.float32) / 2. # (-1, max_distance)
        normed_distance = self.interpolated_gather_nd(distance, new_indices)

        return K.reshape(normed_distance, original_shape)