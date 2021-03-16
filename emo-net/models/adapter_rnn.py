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
from .attention import SeqSelfAttention, SeqWeightedAttention

kernel_regularizer = tf.keras.regularizers.l2(1e-6)
rnn_regularizer = tf.keras.regularizers.L1L2(1e-6)


class RNNWithAdapters(object):
    def __init__(self,
                 input_dims,
                 hidden_size=512,
                 learnall=True,
                 dropout=0.2,
                 layers=2,
                 input_projection_factor=1,
                 adapter_projection_factor=2,
                 tasks=[],
                 recurrent_cell='lstm',
                 bidirectional=False,
                 input_projection=True,
                 input_bn=False,
                 downpool=None,
                 share_feature_layer=False,
                 share_attention=False,
                 use_attention=True,
                 **kwargs):

        self.hidden_size = hidden_size
        self.learnall = learnall
        self.dropout = dropout
        self.tasks = tasks
        self.input_dims = input_dims
        self.adapter_projection_factor = adapter_projection_factor
        self.input_projection = input_projection
        self.input_projection_factor = input_projection_factor
        self.tasks = tasks
        self.recurrent_cell = recurrent_cell
        self.layers = layers
        self.bidirectional = bidirectional
        self.input_bn = input_bn
        self.cnn_input = len(input_dims) > 3
        self.share_feature_layer = share_feature_layer
        self.use_attention = use_attention
        self.share_attention = share_attention
        if self.cnn_input:  # cnn feature extractor
            feature_dims = input_dims[1] * input_dims[3]
        else:
            feature_dims = input_dims[-1]
        self.reorder_dims = tf.keras.layers.Permute((2, 1, 3))
        if downpool is not None:
            self.downpool = tf.keras.layers.AveragePooling1D(
                pool_size=downpool, strides=downpool, padding='same', name='rnn_downpool')
        else:
            self.downpool = None
        self.reshape = tf.keras.layers.Reshape(target_shape=(-1,
                                                             feature_dims))
        if self.input_bn:
            self.input_bns = {task: tf.keras.layers.BatchNormalization(
                trainable=True, name=f'{task}_rnn_input_bn') for task in tasks}
            if not self.input_bns:
                self.core_input_bn = tf.keras.layers.BatchNormalization(
                    trainable=True, name=f'core_rnn_input_bn')
        self.projection = tf.keras.layers.Dense(feature_dims // self.input_projection_factor,
                                                activation=None,
                                                trainable=learnall,
                                                kernel_regularizer=kernel_regularizer)
        self.adapter_hidden_size = self.hidden_size * \
            2 if self.bidirectional else self.hidden_size
        self.rnns = []
        self.selfattentions = []
        self.selfattention = []
        self.adapters = []
        for i in range(self.layers):
            rnn = tf.keras.layers.GRU(self.hidden_size,
                                      dropout=self.dropout,
                                      return_sequences=True,
                                      trainable=learnall, kernel_regularizer=rnn_regularizer) if self.recurrent_cell.lower() == 'gru' else tf.keras.layers.LSTM(self.hidden_size,
                                                                                                                                                                dropout=self.dropout,
                                                                                                                                                                return_sequences=True,
                                                                                                                                                                trainable=learnall, kernel_regularizer=rnn_regularizer)
            if self.bidirectional:
                rnn = tf.keras.layers.Bidirectional(rnn)
            self.rnns.append(rnn)
            self.adapters.append({
                task: RNNAdapter(self.adapter_hidden_size, self.adapter_projection_factor,
                                 task, i)
                for task in tasks
            })
            if i < self.layers - 1: 
                self.selfattentions.append({task: SeqSelfAttention(
                    attention_activation='sigmoid',
                    kernel_regularizer=kernel_regularizer,
                    use_attention_bias=False,
                    trainable=True,
                    name=f'{task}_self_attention_{i}') for task in tasks})
                self.selfattention.append(SeqSelfAttention(
                    attention_activation='sigmoid',
                    kernel_regularizer=kernel_regularizer,
                    use_attention_bias=False,
                    trainable=learnall,
                    name=f'core_self_attention_{i}'))

        self.add = tf.keras.layers.Add()


        self.weighted_attentions = {task: SeqWeightedAttention(
            trainable=True, name=f'{task}_seq_weighted_attention') for task in tasks}
        self.weighted_attention = SeqWeightedAttention(
            trainable=learnall, name=f'core_seq_weighted_attention')

    def __call__(self, x, task, mask=None):
        if self.input_bn:
            if task is not None:
                self.input_bns[task](x)
            else:
                self.core_input_bn(x)
        if self.cnn_input:
            x = self.reorder_dims(x)
            x = self.reshape(x)
        if self.downpool is not None:
            x = self.downpool(x)
        #x = self.mask(x)
        if self.input_projection:
            x = self.projection(x)
        for i in range(self.layers):
            x = self.rnns[i](x, mask=mask)
            if task is not None:
                adapter = self.adapters[i][task](x)
                x = self.add([x, adapter])
                if i < self.layers - 1 and self.use_attention: 
                    if self.share_attention:
                        x = self.selfattention[i](x, mask=mask)
                    else:
                        x = self.selfattentions[i][task](x, mask=mask)
            else:
                if i < self.layers - 1 and self.use_attention:
                    x = self.selfattention[i](x, mask=mask)
        if task is not None and not self.share_feature_layer:
            x = self.weighted_attentions[task](x, mask=mask)
        else:
            x = self.weighted_attention(x, mask=mask)
        return x

    def _add_new_task(self, task):
        assert task not in self.adapters, f'Task {task} already exists!'
        for i in range(self.layers):
            self.adapters[i][task] = RNNAdapter(self.adapter_hidden_size,
                                                self.adapter_projection_factor, task,i)
            if i < self.layers - 1:
                self.selfattentions[i][task] = SeqSelfAttention(
            attention_activation='sigmoid',
            kernel_regularizer=kernel_regularizer,
            use_attention_bias=False,
            trainable=True,
            name=f'{task}_self_attention_{i}')
        self.weighted_attentions[task] = SeqWeightedAttention(
            trainable=True, name=f'{task}_seq_weighted_attention')
        if self.input_bn:
            self.input_bns[task] = tf.keras.layers.BatchNormalization(
                trainable=True, name=f'{task}_rnn_input_bn')


class RNNAdapter(object):
    def __init__(self, input_size, downprojection=4, task='IEMOCAP', index=1):
        self.input_size = input_size
        self.downprojection_factor = downprojection
        self.task = task
        self.layer_norm = tf.keras.layers.TimeDistributed(tf.keras.layers.LayerNormalization(),
                                                          name=f'{task}_rnn_adapter_{index}_layer_norm')
        self.downprojection = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
            self.input_size // self.downprojection_factor, activation='relu', use_bias=False, kernel_regularizer=kernel_regularizer),
            name=f'{task}_rnn_adapter_{index}_downprojection')
        self.upprojection = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.input_size, use_bias=False, kernel_regularizer=kernel_regularizer),
                                                            name=f'{task}_rnn_adapter_{index}_upprojection')

    def __call__(self, x):
        x = self.layer_norm(x)
        x = self.downprojection(x)
        x = self.upprojection(x)
        #x = self.selfattention(x)
        return x
