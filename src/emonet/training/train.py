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
tf.random.set_seed(42)

import time
from datetime import datetime
from os.path import join
from sklearn.utils import class_weight
from .losses import *
from .metrics import *
from ..models.build_model import create_multi_task_resnets, create_multi_task_rnn, create_multi_task_networks
from ..data.loader import *
from os import makedirs
import logging
logger = logging.getLogger(__name__)



categorical_loss_map =  {'crossentropy': "categorical_crossentropy", "focal": categorical_focal_loss(
        ), "ordinal": soft_ordinal_categorical_loss}

binary_loss_map =  {'crossentropy': "binary_crossentropy", "focal": binary_focal_loss(
        ), "ordinal": "binary_crossentropy"}

def determine_decay(generator, batch_size):
    if 10000 > len(generator) * batch_size >= 1000:
        decay = 0.0005
    elif 1000 > len(generator) * batch_size >= 500:
        decay = 0.002
    elif len(generator) * batch_size < 500:
        decay = 0.005
    else:
        decay = 1e-6
    return decay


def named_logs(model, logs):
    result = {}
    for l in zip(model.metrics_names, logs):
        result[l[0]] = l[1]
    return result

def __feature_extractor_params_string(feature_extractor, **kwargs):
    if feature_extractor == 'cnn':
        return __cnn_params(**kwargs)
    elif feature_extractor == 'rnn':
        return __rnn_params(**kwargs)
    elif feature_extractor == 'vgg16':
        return __vgg16_params(**kwargs)
    elif feature_extractor == 'fusion':
        return __fusion_params(**kwargs)
    
def __cnn_params(classifier, N, factor, dropout1, dropout2, rnn_dropout, filters, learnall_classifier):
    return f'filters-{filters}-N-{N}_factor-{factor}-do1-{dropout1}-do2-{dropout2}-classifier-{classifier}-learnall_classifier-{learnall_classifier}{"-rd-"+str(rnn_dropout) if classifier == "rnn" else ""}'

def __fusion_params(N, factor, dropout1, dropout2, rnn_dropout, filters, hidden_dim, cell, bidirectional, number_of_layers, down_pool):
    return f'filters-{filters}-N-{N}_factor-{factor}-do1-{dropout1}-do2-{dropout2}-cell-{cell}-bidirectional-{bidirectional}-hidden_dim-{number_of_layers}x{hidden_dim}-rd{rnn_dropout}-downpool-{down_pool}'


def __rnn_params(hidden_dim, cell, bidirectional, dropout, number_of_layers, down_pool, num_mfccs, use_attention, share_attention, input_projection):
    return f'cell-{cell}-bidirectional-{bidirectional}-hidden_dim-{number_of_layers}x{hidden_dim}-do-{dropout}-downpool-{down_pool}-mfccs-{num_mfccs}-attention-{use_attention}-shareAttention-{share_attention}-ip-{input_projection}'

def __vgg16_params(freeze_up_to, classifier, dropout):
    return f'freezeUpTo-{freeze_up_to}-classifier-{classifier}-dropout-{dropout}'


def train_single_task(
    initial_weights='./weights.h5',
    feature_extractor='cnn',
    batch_size=64,
    epochs=50,
    balanced_weights=True,
    window=6,
    num_mels=128,
    task='IEMOCAP-4cl',
    loss='categorical_cross_entropy',
    directory='./EmoSet/wavs/',
    train_csv='train.csv',
    val_csv='val.csv',
    test_csv='test.csv',
    experiment_base_path='./experiments/residual-adapters-emonet',
    random_noise=None,
    learnall=False,
    last_layer_only=False,
    initial_learning_rate=0.1,
    optimizer=tf.keras.optimizers.SGD,
    n_workers=5,
    patience=20,
    mode='adapters',
    input_bn=False,
    share_feature_layer=False,
    individual_weight_decay=False,
    **kwargs):
    if feature_extractor in ['cnn', 'vgg16']:
        variable_duration = False if kwargs['classifier'] == 'avgpool' else True
    else:
        variable_duration = True
    #variable_duration = False
    base_tasks = None
    base_nb_classes = None
    feature_extractor_params = __feature_extractor_params_string(feature_extractor, **kwargs)
    training_params = f'balancedClassWeights-{balanced_weights}-loss-{loss}-optimizer-{optimizer.__name__}-lr-{initial_learning_rate}-bs-{batch_size}-patience-{patience}-random_noise-{random_noise}-numMels-{num_mels}-ib-{input_bn}-sfl-{share_feature_layer}-iwd-{individual_weight_decay}'
    experiment_base_path = f"{join(experiment_base_path, 'single-task', feature_extractor, mode, f'Window-{window}s', feature_extractor_params, training_params, datetime.now().strftime('%d/%m/%Y-%H:%M:%S'))}"
    
    train_generator = AudioDataGenerator(train_csv,
                                         directory,
                                         batch_size=batch_size,
                                         window=window,
                                         shuffle=True,
                                         sr=16000,
                                         time_stretch=None,
                                         pitch_shift=None,
                                         save_dir=None,
                                         val_split=None,
                                         subset='train',
                                         variable_duration=variable_duration)
    val_generator = AudioDataGenerator(val_csv,
                                       directory,
                                       batch_size=batch_size,
                                       window=window,
                                       shuffle=False,
                                       sr=16000,
                                       time_stretch=None,
                                       pitch_shift=None,
                                       save_dir=None,
                                       variable_duration=variable_duration)
    test_generator = AudioDataGenerator(test_csv,
                                        directory,
                                        batch_size=batch_size,
                                        window=window,
                                        shuffle=False,
                                        sr=16000,
                                        time_stretch=None,
                                        pitch_shift=None,
                                        save_dir=None,
                                        variable_duration=variable_duration)
    val_dataset = val_generator.tf_dataset()
    test_dataset = test_generator.tf_dataset()

    decay = determine_decay(train_generator, batch_size)

    if initial_weights is not None and mode == 'adapters':
        new_tasks = [task]
        new_nb_classes = [len(set(train_generator.classes))]
        new_weight_decays = [decay] if individual_weight_decay else None
        base_weight_decays = None

    else:
        base_tasks = [task]
        base_nb_classes = [len(set(train_generator.classes))]
        base_weight_decays = [decay] if individual_weight_decay else None
        new_tasks = []
        new_nb_classes = []
        new_weight_decays = None

    if balanced_weights:
        class_weights = class_weight.compute_class_weight(
            'balanced', np.unique(train_generator.classes),
            train_generator.classes)
        class_weight_dict = dict(enumerate(class_weights))
        logger.info(f'Class weights: {class_weight_dict}')
    else:
        class_weight_dict = None
        logger.info('Not using class weights.')

    task_base_path = join(experiment_base_path, task, datetime.now().strftime('%d/%m/%Y-%H:%M:%S'))
    weights = join(task_base_path, "weights_" + task + ".h5")

    
    x, _ = train_generator[0]
    if not variable_duration:
        init = x.shape[1:]
    else:
        init = (None, )
    models, shared_model = create_multi_task_networks(
        init,
        feature_extractor=feature_extractor,
        initial_weights=initial_weights,
        num_mels=num_mels,
        mode=mode,
        base_nb_classes=base_nb_classes,
        base_weight_decays=base_weight_decays,
        learnall=learnall,
        base_tasks=base_tasks,
        new_tasks=new_tasks,
        new_nb_classes=new_nb_classes,
        new_weight_decays=new_weight_decays,
        random_noise=random_noise,
        input_bn=input_bn,
        share_feature_layer=share_feature_layer,
        **kwargs)
    model = models[task]
    #model.load_weights(initial_weights, by_name=True)
    if last_layer_only:
        for layer in model.layers[:-5]:
            layer.trainable = False
    model.summary()
    #print(model.non_trainable_weights)
    #model.load_weights(initial_weights, by_name=True)

    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=join(task_base_path, 'log'),
                             histogram_freq=0,
                             write_graph=True)
    #hpCallback = hp.KerasCallback(join(task_base_path, 'log', 'hparam_tuning'), hparams)
    mc = tf.keras.callbacks.ModelCheckpoint(weights,
                         monitor='val_recall/macro/validation',
                         verbose=1,
                         save_best_only=True,
                         save_weights_only=False,
                         mode='max',
                         period=1)
    metric_callback = ClassificationMetricCallback(
        validation_generator=val_dataset.prefetch(tf.data.experimental.AUTOTUNE), dataset_name=task, labels=val_generator.class_indices, true=val_generator.categorical_classes)
    metric_callback_test = ClassificationMetricCallback(
        validation_generator=test_dataset.prefetch(tf.data.experimental.AUTOTUNE), dataset_name=task, partition='test', labels=test_generator.class_indices, true=test_generator.categorical_classes)

    lrs = [
        initial_learning_rate, initial_learning_rate * 0.1,
        initial_learning_rate * 0.01
    ]
    makedirs(task_base_path, exist_ok=True)
    stopped_epoch = 0
    best = 0
    patience = patience
    load = False
    loss_string = loss
    if len(set(train_generator.classes)) == 2:
        loss = binary_loss_map[loss_string]
    else:
        loss = categorical_loss_map[loss_string]
        if loss_string == 'ordinal':
            loss = loss(n_classes=len(set(train_generator.classes)))
    for i, lr in enumerate(lrs):
        if optimizer.__name__ == tf.keras.optimizers.SGD.__name__:
            opt = optimizer(learning_rate=lr, decay=decay if not individual_weight_decay else 1e-6, momentum=0.9, nesterov=False)
        else:
            opt = optimizer(learning_rate=lr)
        model.compile(loss=loss, optimizer=opt, metrics=["acc"], experimental_run_tf_function=False)

        if load:
            model.load_weights(weights)
            logger.info("Model loaded.")
        early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_recall/macro/validation',
                                      min_delta=0.005,
                                      patience=patience,
                                      verbose=1,
                                      mode='max',
                                      restore_best_weights=False,
                                      baseline=best)
        model.fit(train_generator.tf_dataset().prefetch(tf.data.experimental.AUTOTUNE),
                            validation_data=val_generator.tf_dataset(),
                            epochs=epochs,
                            workers=n_workers // 2,
                            initial_epoch=stopped_epoch,
                            class_weight=class_weight_dict,
                            # use_multiprocessing=True,
                            # max_queue_size=n_workers * 2,
                            verbose=2,
                            callbacks=[
                                metric_callback, metric_callback_test,
                                early_stopper, tbCallBack, mc
                            ])
        load = True
        stopped_epoch = early_stopper.stopped_epoch
        best = early_stopper.best


def train_multi_task(
    batch_size=64,
    epochs=50,
    balanced_weights=True,
    feature_extractor='cnn',
    window=2,
    num_mels=128,
    mode='adapters',
    initial_learning_rate=0.1,
    tasks=[
        "AirplaneBehaviourCorpus", "AngerDetection", "BurmeseEmotionalSpeech",
        "CASIA", "ChineseVocalEmotions", "DanishEmotionalSpeech", "DEMoS",
        "EA-ACT", "EA-BMW", "EA-WSJ", "EMO-DB", "EmoFilm", "EmotiW-2014",
        "ENTERFACE", "EU-EmoSS", "FAU_AIBO", "GEMEP", "GVESS",
        "MandarinEmotionalSpeech", "MELD", "PPMK-EMO", "SIMIS", "SMARTKOM",
        "SUSAS", "TurkishEmoBUEE"
    ],
    loss='crossentropy',
    directory='./EmoSet/wavs-reordered/',
    experiment_base_path='./experiments/residual-adapters-emonet',
    multi_task_setup='./EmoSet/multiTaskSetup/',
    steps_per_epoch=20,
    optimizer=tf.keras.optimizers.SGD,
    random_noise=None,
    input_bn=False,
    share_feature_layer=False,
    individual_weight_decay=False,
    **kwargs):
    if feature_extractor == 'cnn':
        variable_duration = False if kwargs['classifier'] == 'avgpool' else True
    else:
        variable_duration = True
    feature_extractor_params = __feature_extractor_params_string(feature_extractor, **kwargs)
    training_params = f'balancedClassWeights-{balanced_weights}-loss-{loss}-optimizer-{optimizer.__name__}-lr-{initial_learning_rate}-bs-{batch_size}-epochs-{epochs}-spe-{steps_per_epoch}-random_noise-{random_noise}-numMels-{num_mels}-ib-{input_bn}-sfl-{share_feature_layer}-iwd-{individual_weight_decay}'
    experiment_base_path = f"{join(experiment_base_path, 'multi-task', '-'.join(map(lambda x: x[:4], tasks)), feature_extractor, mode, f'Window-{window}s', feature_extractor_params, training_params, datetime.now().strftime('%d/%m/%Y-%H:%M:%S'))}"
    
    
    train_generators = [
        AudioDataGenerator(f'{multi_task_setup}/{task}/train.csv',
                           directory,
                           batch_size=batch_size,
                           window=window,
                           shuffle=True,
                           sr=16000,
                           time_stretch=None,
                           pitch_shift=None,
                           variable_duration=variable_duration,
                           save_dir=None,
                           val_split=None,
                           subset='train') for task in tasks
    ]
    val_generators = [
        AudioDataGenerator(f'{multi_task_setup}/{task}/val.csv',
                           directory,
                           batch_size=batch_size,
                           window=window,
                           shuffle=False,
                           sr=16000,
                           time_stretch=None,
                           variable_duration=variable_duration,
                           pitch_shift=None,
                           save_dir=None) for task in tasks
    ]
    test_generators = [
        AudioDataGenerator(f'{multi_task_setup}/{task}/test.csv',
                           directory,
                           batch_size=batch_size,
                           window=window,
                           shuffle=False,
                           sr=16000,
                           time_stretch=None,
                           variable_duration=variable_duration,
                           pitch_shift=None,
                           save_dir=None) for task in tasks
    ]
    
    train_datasets = tuple(gen.tf_dataset().repeat() for gen in train_generators)
    val_datasets = tuple(gen.tf_dataset() for gen in val_generators)
    test_datasets = tuple(gen.tf_dataset() for gen in test_generators)



    if balanced_weights:
        class_weights = [
            class_weight.compute_class_weight('balanced', np.unique(t.classes),
                                              t.classes)
            for t in train_generators
        ]
        class_weight_dicts = [dict(enumerate(cw)) for cw in class_weights]
        logger.info(f'Class weights: {class_weight_dicts}')
    else:
        class_weight_dicts = [None] * len(tasks)
        logger.info('Not using class weights.')

    task_base_paths = [join(experiment_base_path, task) for task in tasks]
    weight_paths = [
        join(task_base_path, "weights_" + task + ".h5")
        for task_base_path, task in zip(task_base_paths, tasks)
    ]

    tbCallBacks = [
        tf.keras.callbacks.TensorBoard(log_dir=join(task_base_path, 'log'),
                    histogram_freq=0,
                    write_graph=True) for task_base_path in task_base_paths
    ]

    metric_callbacks = [
        ClassificationMetricCallback(validation_generator=val_dataset,
                                     period=1, dataset_name=task, labels=val_generator.class_indices)
        for val_dataset, val_generator, task in zip(val_datasets, val_generators, tasks)
    ]
    metric_callbacks_test = [
        ClassificationMetricCallback(validation_generator=test_dataset,
                                     partition='test',
                                     period=1,
                                     dataset_name=task,
                                     labels=test_generator.class_indices)
        for test_dataset, test_generator, task in zip(test_datasets, test_generators, tasks)
    ]
    decays = [determine_decay(tg, batch_size) for tg in train_generators]

    #steps_per_epoch = 10
    x, _ = train_generators[0][0]
    if not variable_duration:
        init = x.shape[1:]
    else:
        init = (None, )

    lrs = [
        initial_learning_rate, initial_learning_rate * 0.1,
        initial_learning_rate * 0.01
    ]
    nb_classes = [len(tg.class_indices) for tg in train_generators]
    models, shared_model = create_multi_task_networks(
        init,
        feature_extractor=feature_extractor,
        mode=mode,
        num_mels=num_mels,
        base_nb_classes=nb_classes,
        learnall=True,
        base_tasks=tasks,
        base_weight_decays=decays if individual_weight_decay else None,
        random_noise=random_noise,
        input_bn=input_bn,
        share_feature_layer=share_feature_layer,
        **kwargs)
    shared_model.summary()
    for i, t in enumerate(tasks):
        tbCallBacks[i].set_model(models[t])
        metric_callbacks[i].set_model(models[t])
        metric_callbacks_test[i].set_model(models[t])

    max_steps= epochs * steps_per_epoch
    loss_string = loss
    for step, lr in enumerate(lrs):
        for i, batch in tqdm(tf.data.Dataset.zip(train_datasets).enumerate().prefetch(1), total=max_steps):
            if i >= max_steps:
                break
            for t, task in enumerate(tasks):
                model = models[task]
                if i == 0:  # reset learning rate
                    if len(set(train_generators[t].classes)) == 2:
                        loss = binary_loss_map[loss_string]
                    else:
                        loss = categorical_loss_map[loss_string]
                        if loss_string == 'ordinal':
                            loss = loss(n_classes=len(set(train_generators[t].classes)))
                    if optimizer.__name__ == tf.keras.optimizers.SGD.__name__:
                        opt = optimizer(lr=lr,
                                        decay=decays[t] if not individual_weight_decay else 1e-6,
                                        momentum=0.9,
                                        nesterov=False)
                    else:
                        opt = optimizer(lr=lr)
                    model.compile(loss=loss, optimizer=opt, metrics=["acc"])
                # if i % len(train_generators[t]) == 0:
                #     train_generators[t].on_epoch_end()
                logs = model.train_on_batch(*batch[t])

                named_l = named_logs(model, logs)
                # loss = named_l["loss"]
                # logger.info(f'Step {i}: loss {loss} ({task})')
                logger.debug(f'i % steps_per_epoch: {i%steps_per_epoch}')
                if i % steps_per_epoch == 0:
                    logger.debug('In epoch end')
                    metric_callbacks[t].on_epoch_end(
                        i // steps_per_epoch + step * epochs, named_l)
                    metric_callbacks_test[t].on_epoch_end(
                        i // steps_per_epoch + step * epochs, named_l)
                    model.save(weight_paths[t])
                    tbCallBacks[t].on_epoch_end(
                        i // steps_per_epoch + step * epochs, named_l)
        shared_model.save(join(experiment_base_path, 'shared_model.h5'))
