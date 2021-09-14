#                    EmoNet
# ==============================================================================
# Copyright (C) 2021 Maurice Gerczuk, Shahin Amiriparian,
# Sandra Ottl, Björn Schuller: University of Augsburg. All Rights Reserved.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================
import tensorflow as tf
import tensorflow.keras.backend as K


class Attention2Dtanh(tf.keras.layers.Layer):
    def __init__(self, lmbda=0.3, mlp_units=256, **kwargs):
        super(Attention2Dtanh, self).__init__(**kwargs)
        self.mlp_units = mlp_units
        self.tanh = tf.keras.layers.Activation('tanh')
        self.lmbda = lmbda

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.mlp_units),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='W')
        self.b = self.add_weight(shape=(self.mlp_units, ),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='b')
        self.u = self.add_weight(shape=(input_shape[-1], ),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='u')
        self.flatten = tf.keras.layers.Reshape(target_shape=(-1,
                                                             input_shape[-1]))
        super(Attention2Dtanh, self).build(input_shape)

    def call(self, inputs):
        flat_input = self.flatten(inputs)
        x = tf.matmul(flat_input, self.w) + self.b
        x = self.tanh(x)
        e = tf.tensordot(self.u, x, axes=[[0], [-1]]) * self.lmbda
        a = tf.nn.softmax(e, axis=-1)
        weighted_sum = tf.reduce_sum(tf.expand_dims(a, -1) * flat_input,
                                     axis=1)
        return weighted_sum

    def get_config(self):
        config = {'lmbda': self.lmbda, 'mlp_units': self.mlp_units}
        base_config = super(Attention2Dtanh, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
