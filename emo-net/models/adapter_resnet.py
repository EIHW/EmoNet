import tensorflow as tf

kernel_regularizer = tf.keras.regularizers.l2(1e-6)

channel_axis = -1

class BasicBlock(object):
    def __init__(self,
                 filters,
                 factor,
                 strides=2,
                 dropout1=0,
                 dropout2=0,
                 shortcut=True,
                 learnall=True,
                 tasks=['IEMOCAP-4cl'],
                 weight_decays=None,
                 **kwargs):
        self.filters = filters
        self.factor = factor
        self.strides = strides
        self.dropout1 = tf.keras.layers.Dropout(dropout1)
        self.dropout2 = tf.keras.layers.Dropout(dropout2)
        self.shortcut = shortcut
        self.learnall = learnall
        self.tasks = weight_decays if weight_decays is not None else [1e-6]*len(self.tasks)
        self.weight_decays = weight_decays
        self.conv_task1 = ConvTasks(filters,
                                                        factor,
                                                        strides=strides,
                                                        learnall=learnall,
                                                        dropout=dropout1,
                                                        tasks=tasks,
                                                        weight_decays=self.weight_decays,
                                                        **kwargs)
        self.conv_task2 = ConvTasks(filters,
                                                        factor,
                                                        strides=1,
                                                        learnall=learnall,
                                                        dropout=dropout2,
                                                        tasks=tasks,
                                                        weight_decays=self.weight_decays,
                                                        **kwargs)

        self.relu = tf.keras.layers.Activation('relu')
        if self.shortcut:
            self.avg_pool = tf.keras.layers.AveragePooling2D((2, 2), padding='same')
            self.lmbda = tf.keras.layers.Lambda(lambda x: x * 0)
        self.add = tf.keras.layers.Add()

    def __call__(self, input_tensor, task):
        residual = input_tensor
        x = self.conv_task1(input_tensor, task=task)
        x = self.relu(x)
        x = self.conv_task2(x, task=task)
        if self.shortcut:
            residual = self.avg_pool(residual)
            residual0 = self.lmbda(residual)
            residual = tf.keras.layers.concatenate([residual, residual0], axis=-1)
        x = self.add([residual, x])
        x = self.relu(x)
        return x
    
    def _add_new_task(self, task, weight_decay=1e-6):
        self.conv_task1._add_new_task(task, weight_decay=weight_decay)
        self.conv_task2._add_new_task(task, weight_decay=weight_decay)
        
        
class ConvTasks(object):
    def __init__(self,
                 filters,
                 factor=1,
                 strides=1,
                 learnall=True,
                 dropout=0,
                 tasks=['IEMOCAP-4cl', 'GEMEP'],
                 weight_decays=None,
                 reuse_batchnorm=False,
                 **kwargs):
        self.filters = filters
        self.factor = factor
        self.strides = strides
        self.learnall = learnall
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.tasks = tasks
        self.weight_decays = weight_decays if weight_decays is not None else [1e-6]*len(self.tasks)
        self.reuse_batchnorm = reuse_batchnorm

        # shared parameters
        self.conv2d = tf.keras.layers.Convolution2D(self.filters * self.factor, (3, 3),
                                    strides=self.strides,
                                    padding='same',
                                    kernel_initializer='he_normal',
                                    use_bias=False,
                                    trainable=self.learnall,
                                    kernel_regularizer=kernel_regularizer)

        # task specificparameters
        self.res_adapts = {}
        self.add = tf.keras.layers.Add()
        self.bns = {}
        self.core_bn = tf.keras.layers.BatchNormalization(
            axis=channel_axis,
            name=f'core_{self.conv2d.name}_batch_normalization')
        for task, weight_decay in zip(self.tasks, self.weight_decays):
            self._add_new_task(task, weight_decay=weight_decay)

    def __call__(self, input_tensor, task):
        in_t = input_tensor
        if task is None:
            in_t = self.dropout(in_t) 
        x = self.conv2d(in_t)
        if task is not None:
            adapter_in = self.dropout(in_t)
            res_adapt = self.res_adapts[task](adapter_in)
            x = self.add([x, res_adapt])
        if self.reuse_batchnorm or task is None:
            x = self.core_bn(x)
        else:
            x = self.bns[task](x)
        return x

    def _add_new_task(self, task, weight_decay=1e-6):
        assert task not in self.bns, 'Task already exists!'
        self.res_adapts[task] = tf.keras.layers.Convolution2D(
            self.filters * self.factor, (1, 1),
            padding='valid',
            kernel_initializer='he_normal',
            strides=self.strides,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            name=f'{task}_{self.conv2d.name}_adapter')
        
        if not self.reuse_batchnorm:
            self.bns[task] = tf.keras.layers.BatchNormalization(
                axis=channel_axis,
                name=f'{task}_{self.conv2d.name}_batch_normalization')


class ResNet(object):
    def __init__(self,
                 filters=32,
                 factor=1,
                 N=2,
                 verbose=1,
                 learnall=True,
                 dropout1=0,
                 dropout2=0,
                 tasks=['IEMOCAP-4cl', 'GEMEP'],
                 weight_decays=None,
                 reuse_batchnorm=False,
                 input_bn=False):
        self.filters = filters
        self.factor = factor
        self.N = N
        self.learnall = learnall
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.tasks = tasks
        self.weight_decays = weight_decays if weight_decays is not None else [1e-6]*len(self.tasks)
        self.reuse_batchnorm = reuse_batchnorm
        self.input_bn = input_bn
        if self.input_bn:
            self.input_core_bn = tf.keras.layers.BatchNormalization(axis=channel_axis, name=f'core_input_batch_normalization')
            self.input_bns = {
                task: tf.keras.layers.BatchNormalization(axis=channel_axis,
                                        name=f'{task}_input_batch_normalization')
                for task in self.tasks
            }
        
        # conv blocks
        self.pre_conv = ConvTasks(filters=self.filters,
                                                      factor=factor,
                                                      strides=1,
                                                      learnall=learnall,
                                                      tasks=self.tasks,
                                                      weight_decays=self.weight_decays,
                                                      reuse_batchnorm=reuse_batchnorm)
        self.blocks = []
        self.nb_conv = 1
        for i in range(1, 4):
            block = BasicBlock(self.filters * (2**i),
                                               self.factor,
                                               strides=2,
                                               dropout1=self.dropout1,
                                               dropout2=self.dropout2,
                                               shortcut=True,
                                               learnall=self.learnall,
                                               tasks=self.tasks,
                                               weight_decays=self.weight_decays,
                                               reuse_batchnorm=reuse_batchnorm)
            self.blocks.append(block)
            for j in range(N - 1):
                block = BasicBlock(filters=self.filters *
                                                   (2**i),
                                                   factor=self.factor,
                                                   strides=1,
                                                   dropout1=self.dropout1,
                                                   dropout2=self.dropout2,
                                                   shortcut=False,
                                                   learnall=self.learnall,
                                                   tasks=self.tasks,
                                                   weight_decays=self.weight_decays,
                                                   reuse_batchnorm=reuse_batchnorm)
                self.blocks.append(block)
                self.nb_conv += 2
        self.nb_conv += 6

        # bns and relus
        self.relu = tf.keras.layers.Activation('relu')
        self.bns = {
            task: tf.keras.layers.BatchNormalization(axis=channel_axis,
                                     name=f'{task}_final_batch_normalization')
            for task in self.tasks
        }
        self.core_bn = tf.keras.layers.BatchNormalization(axis=channel_axis,
                                     name=f'core_final_batch_normalization')

    def _add_new_task(self, task, weight_decay=1e-6):
        assert task not in self.bns, f'Task {task} already exists!'
        self.pre_conv._add_new_task(task, weight_decay=weight_decay)
        for block in self.blocks:
            block._add_new_task(task, weight_decay=weight_decay)
        if not self.reuse_batchnorm:
            self.bns[task] = tf.keras.layers.BatchNormalization(axis=channel_axis, name=f'{task}_final_batch_normalization')
            if self.input_bn:
                self.input_bns[task] = tf.keras.layers.BatchNormalization(axis=channel_axis, name=f'{task}_input_batch_normalization')

    def __call__(self, input_tensor, task):
        if self.input_bn:
            if task is None or self.reuse_batchnorm:
                x = self.input_core_bn(input_tensor)
            else:
                x = self.input_bns[task](input_tensor)
        else:
            x = input_tensor
        x = self.pre_conv(x, task=task)
        for block in self.blocks:
            x = block(x, task=task)
        if task is None or self.reuse_batchnorm:
            x = self.core_bn(x)
        else:
            x = self.bns[task](x)
        x = self.relu(x)
        return x