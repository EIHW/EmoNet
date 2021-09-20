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
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import click
import logging
import logging.config

from .training.train import train_single_task, train_multi_task
from .training.evaluate import evaluate
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta


OPTIMIZERS = {
    'sgd': SGD,
    'rmsprop': RMSprop,
    'adam': Adam,
    'adadelta': Adadelta
}


class OptionEatAll(click.Option):
    def __init__(self, *args, **kwargs):
        self.save_other_options = kwargs.pop('save_other_options', True)
        nargs = kwargs.pop('nargs', -1)
        assert nargs == -1, 'nargs, if set, must be -1 not {}'.format(nargs)
        super(OptionEatAll, self).__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._eat_all_parser = None

    def add_to_parser(self, parser, ctx):
        def parser_process(value, state):
            # method to hook to the parser.process
            done = False
            value = [value]
            if self.save_other_options:
                # grab everything up to the next option
                while state.rargs and not done:
                    for prefix in self._eat_all_parser.prefixes:
                        if state.rargs[0].startswith(prefix):
                            done = True
                    if not done:
                        value.append(state.rargs.pop(0))
            else:
                # grab everything remaining
                value += state.rargs
                state.rargs[:] = []
            value = tuple(value)

            # call the actual process
            self._previous_parser_process(value, state)

        retval = super(OptionEatAll, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(
                name)
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break
        return retval


@click.group()
@click.option('-v', '--verbose', count=True)
def cli(verbose):
    click.echo('Verbosity: %s' % verbose)
    log_levels = ['ERROR', 'INFO', 'DEBUG']
    verbose = min(2, verbose)
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,  # this fixes the problem
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': log_levels[verbose],
                'class': 'logging.StreamHandler',
                'formatter': 'standard'
            },
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': log_levels[verbose],
                'propagate': True
            }
        }
    })

@cli.group()
def cnn():
    pass

# @cli.group()
# def rnn():
#     pass

@cli.group()
def vgg16():
    pass

# @cli.group()
# def fusion():
#     pass



# @fusion.command(name='multi-task')
@click.option('-dp',
              '--data-path',
              required=True,
              help='Directory of data files.',
              type=click.Path(file_okay=False))
@click.option('-mts',
              '--multi-task-setup',
              required=True,
              help='Directory with the setup csvs for each task in a separate folder.',
              type=click.Path(file_okay=False))
@click.option('-t',
              '--tasks',
              required=True,
              help='Names of the tasks that are trained.',
              cls=OptionEatAll)
@click.option(
    '-bs',
    '--batch-size',
    type=int,
    help='Define batch size.',
    default=64,
)
@click.option(
    '-nm',
    '--num-mels',
    type=int,
    help='Number of mel bands in spectrogram.',
    default=128,
)
@click.option(
    '-e',
    '--epochs',
    type=int,
    help='Define max number of training epochs.',
    default=50,
)
@click.option(
    '-spe',
    '--steps-per-epoch',
    type=int,
    help=
    'Number of training steps for each artificial epoch.',
    default=20
)

@click.option('-bw',
              '--balanced-weights',
              help='Automatically set balanced class weights.',
              is_flag=True)
@click.option(
    '-lr',
    '--learning-rate',
    type=click.FloatRange(0, 1),
    default=0.1,
    help='Initial earning rate for optimizer.',
)
@click.option(
    '-do',
    '--dropout',
    type=click.FloatRange(0, 1),
    default=(0, 0.5),
    nargs=2,
    help=
    'Dropout for the two positions (after first and second convolution of each block).'
)
@click.option(
    '-rd',
    '--recurrent-dropout',
    type=click.FloatRange(0, 1),
    default=0.2,
    nargs=1,
    help=
    'Dropout for the rnn classifier, if used.'
)
@click.option(
    '-nw',
    '--n-workers',
    type=int,
    help='Number of training worker processes.',
    default=5,
)
@click.option(
    '-ebp',
    '--experiment-base-path',
    type=click.Path(writable=True, readable=True),
    help='Basepath where logs and checkpoints should be stored.',
    default='./residual-adapters',
)
@click.option('-o',
              '--optimizer',
              type=click.Choice(OPTIMIZERS.keys()),
              help='Optimizer used for training.',
              default='sgd')
@click.option('-N',
              '--number-of-resnet-blocks',
              help='Number of convolutional blocks in the ResNet layers.',
              type=int,
              default=2)
@click.option('-wf',
              '--widen-factor',
              type=int,
              help='Widen factor of wide ResNet',
              default=1)
@click.option('-nf',
              '--number-of-filters',
              help='Number of filters in first convolutional block.',
              type=int,
              default=32)
@click.option('-w',
              '--window',
              type=float,
              help='Window size in seconds.',
              default=None)
@click.option('-rn',
              '--random-noise',
              type=float,
              help='Power of random gaussian noise to add to input spectrograms for augmentation.',
              default=None)
@click.option('-l',
              '--loss',
              type=click.Choice(['crossentropy', 'focal', 'ordinal']),
              help='Classification loss.',
              default='crossentropy')
@click.option('-dp',
              '--down-pool',
              help='Downpooling factor for input MFCCs.',
              type=int,
              default=1)
@click.option('-hd',
              '--hidden-dim',
              help='Dimension of hidden rnn layers.',
              type=int,
              default=512)
@click.option('-nl',
              '--number-of-layers',
              help='Number of hidden rnn layers.',
              type=int,
              default=2)
@click.option('-ct',
              '--cell-type',
              type=click.Choice(['lstm', 'gru']),
              default='lstm')
@click.option('-ib',
              '--input-bn',
              help='Introduce BatchNormalization directly after input.',
              is_flag=True)
@click.option('-sfl',
              '--share-feature-layer',
              help='Share the feature layer (weighted attention of deep features) between tasks.',
              is_flag=True)
@click.option('--bidirectional/--unidirectional', default=False)
def fusion_multi_task(data_path,
               multi_task_setup,
                tasks,
                batch_size=64,
                num_mels=128,
                epochs=50,
                experiment_base_path='./experiments/residual-adapters',
                optimizer='sgd',
                learning_rate=0.1,
                balanced_weights=False,
                number_of_filters=32,
                recurrent_dropout=0.2,
                number_of_resnet_blocks=2,
                widen_factor=1,
                down_pool=1,
                number_of_layers=2,
                hidden_dim=256,
                cell_type='lstm',
                bidirectional=False,
                n_workers=5,
                dropout=(0, 0.5),
                loss='crossentropy',
                window=None,
                random_noise=None,
                steps_per_epoch=20,
                input_bn=False,
                share_feature_layer=False):
    train_multi_task(
    batch_size=batch_size,
    feature_extractor='fusion',
    epochs=epochs,
    balanced_weights=balanced_weights,
    window=window,
    num_mels=num_mels,
    N=number_of_resnet_blocks,
    factor=widen_factor,
    filters=number_of_filters,
    dropout1=dropout[0],
    dropout2=dropout[1],
    rnn_dropout=recurrent_dropout,
    down_pool=down_pool,
    hidden_dim=hidden_dim,
    number_of_layers=number_of_layers,
    cell=cell_type,
    bidirectional=bidirectional,
    tasks=tasks,
    loss=loss,
    directory=data_path,
    experiment_base_path=experiment_base_path,
    multi_task_setup=multi_task_setup,
    random_noise=random_noise,
    steps_per_epoch=steps_per_epoch,
    optimizer=OPTIMIZERS[optimizer],
    input_bn=input_bn,
    share_feature_layer=share_feature_layer)
    
# @fusion.command(name='single-task')
@click.option('-dp',
              '--data-path',
              required=True,
              help='Directory of data files.',
              type=click.Path(file_okay=False))
@click.option('-t',
              '--task',
              required=True,
              help='Name of the task that is trained.',
              type=str)
@click.option('-tr',
              '--train-csv',
              nargs=1,
              required=True,
              help='Path to training csv file.',
              type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('-v',
              '--val-csv',
              required=True,
              help='Path to validation csv file.',
              nargs=1,
              type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('-te',
              '--test-csv',
              required=True,
              help='Path to validation csv file.',
              nargs=1,
              type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    '-bs',
    '--batch-size',
    type=int,
    help='Define batch size.',
    default=64,
)
@click.option(
    '-nm',
    '--num-mels',
    type=int,
    help='Number of mel bands in spectrogram.',
    default=128,
)
@click.option(
    '-e',
    '--epochs',
    type=int,
    help='Define max number of training epochs.',
    default=1000,
)
@click.option(
    '-p',
    '--patience',
    type=int,
    help=
    'Define patience before early stopping / reducing learning rate in epochs.',
    default=20,
)
@click.option(
    '-im',
    '--initial-model',
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help='Initial model for resuming training.',
    default=None,
)
@click.option('-bw',
              '--balanced-weights',
              help='Automatically set balanced class weights.',
              is_flag=True)
@click.option(
    '-lr',
    '--learning-rate',
    type=click.FloatRange(0, 1),
    default=0.1,
    help='Initial earning rate for optimizer.',
)
@click.option(
    '-do',
    '--dropout',
    type=click.FloatRange(0, 1),
    default=(0, 0.5),
    nargs=2,
    help=
    'Dropout for the two positions (after first and second convolution of each block).'
)
@click.option(
    '-rd',
    '--rnn-dropout',
    type=click.FloatRange(0, 1),
    default=0.2,
    help='Dropout for rnn if used as classifier.',
)
@click.option(
    '-ebp',
    '--experiment-base-path',
    type=click.Path(writable=True, readable=True),
    help='Basepath where logs and checkpoints should be stored.',
    default='./residual-adapters',
)
@click.option('-o',
              '--optimizer',
              type=click.Choice(OPTIMIZERS.keys()),
              help='Optimizer used for training.',
              default='sgd')
@click.option('-N',
              '--number-of-resnet-blocks',
              help='Number of convolutional blocks in the ResNet layers.',
              type=int,
              default=2)
@click.option('-nf',
              '--number-of-filters',
              help='Number of filters in first convolutional block.',
              type=int,
              default=32)
@click.option('-wf',
              '--widen-factor',
              type=int,
              help='Widen factor of wide ResNet',
              default=1)
@click.option('-w',
              '--window',
              type=float,
              help='Window size in seconds.',
              default=None)
@click.option('-rn',
              '--random-noise',
              type=float,
              help='Power of random gaussian noise to add to input spectrograms for augmentation.',
              default=None)
@click.option('-l',
              '--loss',
              type=click.Choice(['crossentropy', 'focal', 'ordinal']),
              help='Classification loss.',
              default='crossentropy')
@click.option('-m',
              '--mode',
              type=click.Choice(['scratch', 'adapters', 'last-layer', 'finetune']))
@click.option('-nw',
              '--n-workers',
              help='Number of training workers.',
              type=int,
              default=5)
@click.option('-ib',
              '--input-bn',
              help='Introduce BatchNormalization directly after input.',
              is_flag=True)
@click.option('-sfl',
              '--share-feature-layer',
              help='Share the feature layer (weighted attention of deep features) between tasks.',
              is_flag=True)
@click.option('-dp',
              '--down-pool',
              help='Downpooling factor for input MFCCs.',
              type=int,
              default=1)
@click.option('-hd',
              '--hidden-dim',
              help='Dimension of hidden rnn layers.',
              type=int,
              default=512)
@click.option('-nl',
              '--number-of-layers',
              help='Number of hidden rnn layers.',
              type=int,
              default=2)
@click.option('-ct',
              '--cell-type',
              type=click.Choice(['lstm', 'gru']),
              default='lstm')
@click.option('--bidirectional/--unidirectional', default=False)
def fusion_single_task(data_path,
                task,
                train_csv='train.csv',
                val_csv='val.csv',
                test_csv='test.csv',
                batch_size=64,
                epochs=1000,
                initial_model=None,
                experiment_base_path='./experiments/residual-adapters',
                optimizer='sgd',
                num_mels=128,
                learning_rate=0.1,
                balanced_weights=False,
                number_of_filters=32,
                rnn_dropout=0.2,
                number_of_resnet_blocks=2,
                widen_factor=1,
                down_pool=1,
                number_of_layers=2,
                hidden_dim=256,
                cell_type='lstm',
                bidirectional=False,
                n_workers=5,
                dropout=(0, 0.5),
                loss='crossentropy',
                window=None,
                patience=20,
                mode='scratch',
                random_noise=None,
                input_bn=False,
                share_feature_layer=False):
    learnall = mode == 'scratch' or mode == 'finetune'
    last_layer_only = True if mode == 'last-layer' else False

    train_single_task(initial_weights=initial_model,
                    feature_extractor='fusion',
                    batch_size=batch_size,
                    epochs=epochs,
                    balanced_weights=balanced_weights,
                    window=window,
                    num_mels=num_mels,
                    N=number_of_resnet_blocks,
                    factor=widen_factor,
                    dropout1=dropout[0],
                    dropout2=dropout[1],
                    task=task,
                    filters=number_of_filters,
                    rnn_dropout=rnn_dropout,
                    down_pool=down_pool,
                    number_of_layers=number_of_layers,
                    hidden_dim=hidden_dim,
                    cell=cell_type,
                    bidirectional=bidirectional,
                    mode=mode,
                    loss=loss,
                    directory=data_path,
                    train_csv=train_csv,
                    val_csv=val_csv,
                    test_csv=test_csv,
                    experiment_base_path=experiment_base_path,
                    random_noise=random_noise,
                    learnall=learnall,
                    last_layer_only=last_layer_only,
                    initial_learning_rate=learning_rate,
                    optimizer=OPTIMIZERS[optimizer],
                    patience=patience,
                    n_workers=n_workers,
                    input_bn=input_bn,
                    share_feature_layer=share_feature_layer)


@cnn.command(name='single-task')
@click.option('-dp',
              '--data-path',
              required=True,
              help='Directory of data files.',
              type=click.Path(file_okay=False))
@click.option('-t',
              '--task',
              required=True,
              help='Name of the task that is trained.',
              type=str)
@click.option('-tr',
              '--train-csv',
              nargs=1,
              required=True,
              help='Path to training csv file.',
              type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('-v',
              '--val-csv',
              required=True,
              help='Path to validation csv file.',
              nargs=1,
              type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('-te',
              '--test-csv',
              required=True,
              help='Path to validation csv file.',
              nargs=1,
              type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    '-bs',
    '--batch-size',
    type=int,
    help='Define batch size.',
    default=64,
)
@click.option(
    '-nm',
    '--num-mels',
    type=int,
    help='Number of mel bands in spectrogram.',
    default=128,
)
@click.option(
    '-e',
    '--epochs',
    type=int,
    help='Define max number of training epochs.',
    default=1000,
)
@click.option(
    '-p',
    '--patience',
    type=int,
    help=
    'Define patience before early stopping / reducing learning rate in epochs.',
    default=20,
)
@click.option(
    '-im',
    '--initial-model',
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help='Initial model for resuming training.',
    default=None,
)
@click.option('-bw',
              '--balanced-weights',
              help='Automatically set balanced class weights.',
              is_flag=True)
@click.option(
    '-lr',
    '--learning-rate',
    type=click.FloatRange(0, 1),
    default=0.1,
    help='Initial earning rate for optimizer.',
)
@click.option(
    '-do',
    '--dropout',
    type=click.FloatRange(0, 1),
    default=(0, 0.5),
    nargs=2,
    help=
    'Dropout for the two positions (after first and second convolution of each block).'
)
@click.option(
    '-rd',
    '--rnn-dropout',
    type=click.FloatRange(0, 1),
    default=0.2,
    help='Dropout for rnn if used as classifier.',
)
@click.option(
    '-ebp',
    '--experiment-base-path',
    type=click.Path(writable=True, readable=True),
    help='Basepath where logs and checkpoints should be stored.',
    default='./residual-adapters',
)
@click.option('-o',
              '--optimizer',
              type=click.Choice(OPTIMIZERS.keys()),
              help='Optimizer used for training.',
              default='sgd')
@click.option('-N',
              '--number-of-resnet-blocks',
              help='Number of convolutional blocks in the ResNet layers.',
              type=int,
              default=2)
@click.option('-nf',
              '--number-of-filters',
              help='Number of filters in first convolutional block.',
              type=int,
              default=32)
@click.option('-wf',
              '--widen-factor',
              type=int,
              help='Widen factor of wide ResNet',
              default=1)
@click.option(
    '-c',
    '--classifier',
    type=click.Choice(['avgpool', 'FCNAttention', 'attention', 'attention2d', 'rnn']),
    help=
    'The classification top of the network architeture. Choose between simple pooling + dense layer (needs fixes window size) and fully convolutional attention.',
    default='avgpool')
@click.option('-w',
              '--window',
              type=float,
              help='Window size in seconds.',
              default=None)
@click.option('-rn',
              '--random-noise',
              type=float,
              help='Power of random gaussian noise to add to input spectrograms for augmentation.',
              default=None)
@click.option('-l',
              '--loss',
              type=click.Choice(['crossentropy', 'focal', 'ordinal']),
              help='Classification loss.',
              default='crossentropy')
@click.option('-m',
              '--mode',
              type=click.Choice(['scratch', 'adapters', 'last-layer', 'finetune']))
@click.option('-nw',
              '--n-workers',
              help='Number of training workers.',
              type=int,
              default=5)
@click.option('-ib',
              '--input-bn',
              help='Introduce BatchNormalization directly after input.',
              is_flag=True)
@click.option('-sfl',
              '--share-feature-layer',
              help='Share the feature layer (weighted attention of deep features) between tasks.',
              is_flag=True)
@click.option('-iwd',
              '--individual-weight-decay',
              help='Set weight decay in adapters according to size of training dataset. Smaller datasets will have larger weight decay to keep closer to the pre-trained network.',
              is_flag=True)
def single_task(data_path,
                task,
                train_csv='train.csv',
                val_csv='val.csv',
                test_csv='test.csv',
                batch_size=64,
                epochs=1000,
                initial_model=None,
                experiment_base_path='./experiments/residual-adapters',
                optimizer='sgd',
                num_mels=128,
                learning_rate=0.1,
                balanced_weights=False,
                number_of_filters=32,
                rnn_dropout=0.2,
                number_of_resnet_blocks=2,
                widen_factor=1,
                n_workers=5,
                dropout=(0, 0.5),
                loss='crossentropy',
                window=None,
                classifier='avgpool',
                patience=20,
                mode='scratch',
                random_noise=None,
                input_bn=False,
                share_feature_layer=False,
                individual_weight_decay=False):
    learnall = mode == 'scratch' or mode == 'finetune'
    learnall_classifier = mode == 'scratch' or mode == 'finetune'
    last_layer_only = True if mode == 'last-layer' else False

    train_single_task(initial_weights=initial_model,
                    feature_extractor='cnn',
                     batch_size=batch_size,
                     epochs=epochs,
                     balanced_weights=balanced_weights,
                     window=window,
                     num_mels=num_mels,
                     N=number_of_resnet_blocks,
                     factor=widen_factor,
                     dropout1=dropout[0],
                     dropout2=dropout[1],
                     task=task,
                     filters=number_of_filters,
                     rnn_dropout=rnn_dropout,
                     mode=mode,
                     classifier=classifier,
                     learnall_classifier=learnall_classifier,
                     loss=loss,
                     directory=data_path,
                     train_csv=train_csv,
                     val_csv=val_csv,
                     test_csv=test_csv,
                     experiment_base_path=experiment_base_path,
                     random_noise=random_noise,
                     learnall=learnall,
                     last_layer_only=last_layer_only,
                     initial_learning_rate=learning_rate,
                     optimizer=OPTIMIZERS[optimizer],
                     patience=patience,
                     n_workers=n_workers,
                     input_bn=input_bn,
                     share_feature_layer=share_feature_layer,
                     individual_weight_decay=individual_weight_decay)

@cnn.command(name='evaluate')
@click.option('-dp',
              '--data-path',
              required=True,
              help='Directory of data files.',
              type=click.Path(file_okay=False))
@click.option('-t',
              '--task',
              required=True,
              help='Name of the task that is trained.',
              type=str)
@click.option('-v',
              '--val-csv',
              required=True,
              help='Path to validation csv file.',
              nargs=1,
              type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('-po',
              '--predictions-output',
              required=True,
              help='Path to prediction output.',
              nargs=1,
              type=click.Path(dir_okay=False, readable=True))
@click.option(
    '-bs',
    '--batch-size',
    type=int,
    help='Define batch size.',
    default=64,
)
@click.option(
    '-nm',
    '--num-mels',
    type=int,
    help='Number of mel bands in spectrogram.',
    default=128,
)
@click.option(
    '-im',
    '--initial-model',
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help='Saved model to evaluate.',
    default=None,
    required=True
)
@click.option('-N',
              '--number-of-resnet-blocks',
              help='Number of convolutional blocks in the ResNet layers.',
              type=int,
              default=2)
@click.option('-nf',
              '--number-of-filters',
              help='Number of filters in first convolutional block.',
              type=int,
              default=32)
@click.option('-wf',
              '--widen-factor',
              type=int,
              help='Widen factor of wide ResNet',
              default=1)
@click.option(
    '-c',
    '--classifier',
    type=click.Choice(['avgpool', 'FCNAttention', 'rnn']),
    help=
    'The classification top of the network architeture. Choose between simple pooling + dense layer (needs fixed window size) and fully convolutional attention.',
    default='avgpool')
@click.option('-w',
              '--window',
              type=float,
              help='Window size in seconds.',
              default=None)
@click.option('-ib',
              '--input-bn',
              help='Introduce BatchNormalization directly after input.',
              is_flag=True)
@click.option('-m',
              '--mode',
              type=click.Choice(['scratch', 'adapters', 'last-layer', 'finetune']))
@click.option('-sfl',
              '--share-feature-layer',
              help='Share the feature layer (weighted attention of deep features) between tasks.',
              is_flag=True)
def evaluate_cnn(data_path,
                val_csv='val.csv',
                batch_size=64,
                initial_model=None,
                num_mels=128,
                task="",
                number_of_filters=32,
                number_of_resnet_blocks=2,
                widen_factor=1,
                window=None,
                classifier='avgpool',
                input_bn=False,
                mode='adapters',
                predictions_output='pred.csv',
                share_feature_layer=False):

    evaluate(initial_weights=initial_model,
                    feature_extractor='cnn',
                     batch_size=batch_size,
                     window=window,
                     num_mels=num_mels,
                     N=number_of_resnet_blocks,
                     task=task,
                     factor=widen_factor,
                     mode=mode,
                     filters=number_of_filters,
                     classifier=classifier,
                     directory=data_path,
                     val_csv=val_csv,
                     output=predictions_output,
                     input_bn=input_bn,
                     share_feature_layer=share_feature_layer)


@cnn.command(name='multi-task')
@click.option('-dp',
              '--data-path',
              required=True,
              help='Directory of data files.',
              type=click.Path(file_okay=False))
@click.option('-mts',
              '--multi-task-setup',
              required=True,
              help='Directory with the setup csvs for each task in a separate folder.',
              type=click.Path(file_okay=False))
@click.option('-t',
              '--tasks',
              required=True,
              help='Names of the tasks that are trained.',
              multiple=True)
@click.option(
    '-bs',
    '--batch-size',
    type=int,
    help='Define batch size.',
    default=64,
)
@click.option(
    '-nm',
    '--num-mels',
    type=int,
    help='Number of mel bands in spectrogram.',
    default=128,
)
@click.option(
    '-e',
    '--epochs',
    type=int,
    help='Define max number of training epochs.',
    default=50,
)
@click.option(
    '-spe',
    '--steps-per-epoch',
    type=int,
    help=
    'Number of training steps for each artificial epoch.',
    default=20
)

@click.option('-bw',
              '--balanced-weights',
              help='Automatically set balanced class weights.',
              is_flag=True)
@click.option(
    '-lr',
    '--learning-rate',
    type=click.FloatRange(0, 1),
    default=0.1,
    help='Initial earning rate for optimizer.',
)
@click.option(
    '-do',
    '--dropout',
    type=click.FloatRange(0, 1),
    default=(0, 0.5),
    nargs=2,
    help=
    'Dropout for the two positions (after first and second convolution of each block).'
)
@click.option(
    '-rd',
    '--recurrent-dropout',
    type=click.FloatRange(0, 1),
    default=0.2,
    nargs=1,
    help=
    'Dropout for the rnn classifier, if used.'
)
@click.option(
    '-nw',
    '--n-workers',
    type=int,
    help='Number of training worker processes.',
    default=5,
)
@click.option(
    '-ebp',
    '--experiment-base-path',
    type=click.Path(writable=True, readable=True),
    help='Basepath where logs and checkpoints should be stored.',
    default='./residual-adapters',
)
@click.option('-o',
              '--optimizer',
              type=click.Choice(OPTIMIZERS.keys()),
              help='Optimizer used for training.',
              default='sgd')
@click.option('-N',
              '--number-of-resnet-blocks',
              help='Number of convolutional blocks in the ResNet layers.',
              type=int,
              default=2)
@click.option('-wf',
              '--widen-factor',
              type=int,
              help='Widen factor of wide ResNet',
              default=1)
@click.option('-nf',
              '--number-of-filters',
              help='Number of filters in first convolutional block.',
              type=int,
              default=32)
@click.option(
    '-c',
    '--classifier',
    type=click.Choice(['avgpool', 'FCNAttention', 'rnn']),
    help=
    'The classification top of the network architeture. Choose between simple pooling + dense layer (needs fixes window size) and fully convolutional attention.',
    default='avgpool')
@click.option('-w',
              '--window',
              type=float,
              help='Window size in seconds.',
              default=None)
@click.option('-rn',
              '--random-noise',
              type=float,
              help='Power of random gaussian noise to add to input spectrograms for augmentation.',
              default=None)
@click.option('-l',
              '--loss',
              type=click.Choice(['crossentropy', 'focal', 'ordinal']),
              help='Classification loss.',
              default='crossentropy')
@click.option('-ib',
              '--input-bn',
              help='Introduce BatchNormalization directly after input.',
              is_flag=True)
@click.option('-sfl',
              '--share-feature-layer',
              help='Share the feature layer (weighted attention of deep features) between tasks.',
              is_flag=True)
@click.option('-iwd',
              '--individual-weight-decay',
              help='Set weight decay in adapters according to size of training dataset. Smaller datasets will have larger weight decay to keep closer to the pre-trained network.',
              is_flag=True)
def multi_task(data_path,
               multi_task_setup,
                tasks,
                batch_size=64,
                num_mels=128,
                epochs=50,
                experiment_base_path='./experiments/residual-adapters',
                optimizer='sgd',
                learning_rate=0.1,
                balanced_weights=False,
                number_of_filters=32,
                recurrent_dropout=0.2,
                number_of_resnet_blocks=2,
                widen_factor=1,
                n_workers=5,
                dropout=(0, 0.5),
                loss='crossentropy',
                window=None,
                classifier='avgpool',
                random_noise=None,
                steps_per_epoch=20,
                input_bn=False,
                share_feature_layer=False,
                individual_weight_decay=False):
    train_multi_task(
    batch_size=batch_size,
    epochs=epochs,
    initial_learning_rate=learning_rate,
    balanced_weights=balanced_weights,
    window=window,
    num_mels=num_mels,
    N=number_of_resnet_blocks,
    factor=widen_factor,
    filters=number_of_filters,
    dropout1=dropout[0],
    dropout2=dropout[1],
    rnn_dropout=recurrent_dropout,
    learnall_classifier=True,
    tasks=tasks,
    loss=loss,
    classifier=classifier,
    directory=data_path,
    experiment_base_path=experiment_base_path,
    multi_task_setup=multi_task_setup,
    random_noise=random_noise,
    steps_per_epoch=steps_per_epoch,
    optimizer=OPTIMIZERS[optimizer],
    input_bn=input_bn,
    share_feature_layer=share_feature_layer,
    individual_weight_decay=individual_weight_decay)

@vgg16.command(name='single-task')
@click.option('-dp',
              '--data-path',
              required=True,
              help='Directory of data files.',
              type=click.Path(file_okay=False))
@click.option('-t',
              '--task',
              required=True,
              help='Name of the task that is trained.',
              type=str)
@click.option('-tr',
              '--train-csv',
              nargs=1,
              required=True,
              help='Path to training csv file.',
              type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('-v',
              '--val-csv',
              required=True,
              help='Path to validation csv file.',
              nargs=1,
              type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('-te',
              '--test-csv',
              required=True,
              help='Path to validation csv file.',
              nargs=1,
              type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    '-bs',
    '--batch-size',
    type=int,
    help='Define batch size.',
    default=64,
)
@click.option(
    '-nm',
    '--num-mels',
    type=int,
    help='Number of mel bands in spectrogram.',
    default=128,
)
@click.option(
    '-e',
    '--epochs',
    type=int,
    help='Define max number of training epochs.',
    default=1000,
)
@click.option(
    '-p',
    '--patience',
    type=int,
    help=
    'Define patience before early stopping / reducing learning rate in epochs.',
    default=20,
)
@click.option(
    '-im',
    '--initial-model',
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help='Initial model for resuming training.',
    default=None,
)
@click.option('-bw',
              '--balanced-weights',
              help='Automatically set balanced class weights.',
              is_flag=True)
@click.option(
    '-lr',
    '--learning-rate',
    type=click.FloatRange(0, 1),
    default=0.1,
    help='Initial earning rate for optimizer.',
)
@click.option(
    '-do',
    '--dropout',
    type=click.FloatRange(0, 1),
    default=0.2,
    nargs=1,
    help=
    'Dropout for the two positions (after first and second convolution of each block).'
)
@click.option(
    '-ebp',
    '--experiment-base-path',
    type=click.Path(writable=True, readable=True),
    help='Basepath where logs and checkpoints should be stored.',
    default='./residual-adapters',
)
@click.option('-o',
              '--optimizer',
              type=click.Choice(OPTIMIZERS.keys()),
              help='Optimizer used for training.',
              default='sgd')
@click.option(
    '-c',
    '--classifier',
    type=click.Choice(['avgpool', 'FCNAttention', 'rnn']),
    help=
    'The classification top of the network architeture. Choose between simple pooling + dense layer (needs fixes window size) and fully convolutional attention.',
    default='avgpool')
@click.option('-w',
              '--window',
              type=float,
              help='Window size in seconds.',
              default=None)
@click.option('-rn',
              '--random-noise',
              type=float,
              help='Power of random gaussian noise to add to input spectrograms for augmentation.',
              default=None)
@click.option('-l',
              '--loss',
              type=click.Choice(['crossentropy', 'focal', 'ordinal']),
              help='Classification loss.',
              default='crossentropy')
@click.option('-nw',
              '--n-workers',
              help='Number of training workers.',
              type=int,
              default=5)
@click.option('-ft',
              '--freeze-up-to',
              help='Freeze layers up to index.',
              type=int,
              default=None)
@click.option('-sfl',
              '--share-feature-layer',
              help='Share the feature layer (weighted attention of deep features) between tasks.',
              is_flag=True)
def single_task_vgg(data_path,
                task,
                train_csv='train.csv',
                val_csv='val.csv',
                test_csv='test.csv',
                batch_size=64,
                num_mels=128,
                epochs=1000,
                initial_model=None,
                experiment_base_path='./experiments/residual-adapters',
                optimizer='sgd',
                learning_rate=0.1,
                balanced_weights=False,
                n_workers=5,
                dropout=0.2,
                loss='crossentropy',
                window=None,
                classifier='avgpool',
                patience=20,
                random_noise=None,
                share_feature_layer=False,
                freeze_up_to=None):

    train_single_task(initial_weights=initial_model,
                    feature_extractor='vgg16',
                     batch_size=batch_size,
                     epochs=epochs,
                     balanced_weights=balanced_weights,
                     window=window,
                     num_mels=num_mels,
                     dropout=dropout,
                     task=task,
                     classifier=classifier,
                     loss=loss,
                     directory=data_path,
                     train_csv=train_csv,
                     val_csv=val_csv,
                     test_csv=test_csv,
                     experiment_base_path=experiment_base_path,
                     random_noise=random_noise,
                     initial_learning_rate=learning_rate,
                     optimizer=OPTIMIZERS[optimizer],
                     patience=patience,
                     n_workers=n_workers,
                     freeze_up_to=freeze_up_to,
                     share_feature_layer=share_feature_layer)

@vgg16.command(name='multi-task')
@click.option('-dp',
              '--data-path',
              required=True,
              help='Directory of data files.',
              type=click.Path(file_okay=False))
@click.option('-mts',
              '--multi-task-setup',
              required=True,
              help='Directory with the setup csvs for each task in a separate folder.',
              type=click.Path(file_okay=False))
@click.option('-t',
              '--tasks',
              required=True,
              multiple=True,
              help='Names of the tasks that are trained.')
@click.option(
    '-bs',
    '--batch-size',
    type=int,
    help='Define batch size.',
    default=64,
)
@click.option(
    '-nm',
    '--num-mels',
    type=int,
    help='Number of mel bands in spectrogram.',
    default=128,
)
@click.option(
    '-e',
    '--epochs',
    type=int,
    help='Define max number of training epochs.',
    default=50,
)
@click.option(
    '-spe',
    '--steps-per-epoch',
    type=int,
    help=
    'Number of training steps for each artificial epoch.',
    default=20
)

@click.option('-bw',
              '--balanced-weights',
              help='Automatically set balanced class weights.',
              is_flag=True)
@click.option(
    '-lr',
    '--learning-rate',
    type=click.FloatRange(0, 1),
    default=0.1,
    help='Initial earning rate for optimizer.',
)
@click.option(
    '-do',
    '--dropout',
    type=click.FloatRange(0, 1),
    default=0.2,
    nargs=1,
    help=
    'Dropout for the two positions (after first and second convolution of each block).'
)
@click.option(
    '-nw',
    '--n-workers',
    type=int,
    help='Number of training worker processes.',
    default=5,
)
@click.option(
    '-ebp',
    '--experiment-base-path',
    type=click.Path(writable=True, readable=True),
    help='Basepath where logs and checkpoints should be stored.',
    default='./residual-adapters',
)
@click.option('-o',
              '--optimizer',
              type=click.Choice(OPTIMIZERS.keys()),
              help='Optimizer used for training.',
              default='sgd')
@click.option(
    '-c',
    '--classifier',
    type=click.Choice(['avgpool', 'FCNAttention', 'rnn']),
    help=
    'The classification top of the network architeture. Choose between simple pooling + dense layer (needs fixes window size) and fully convolutional attention.',
    default='avgpool')
@click.option('-w',
              '--window',
              type=float,
              help='Window size in seconds.',
              default=None)
@click.option('-rn',
              '--random-noise',
              type=float,
              help='Power of random gaussian noise to add to input spectrograms for augmentation.',
              default=None)
@click.option('-l',
              '--loss',
              type=click.Choice(['crossentropy', 'focal', 'ordinal']),
              help='Classification loss.',
              default='crossentropy')
@click.option('-sfl',
              '--share-feature-layer',
              help='Share the feature layer (weighted attention of deep features) between tasks.',
              is_flag=True)
def multi_task_vgg(data_path,
               multi_task_setup,
                tasks,
                batch_size=64,
                epochs=50,
                num_mels=128,
                experiment_base_path='./experiments/residual-adapters',
                optimizer='sgd',
                learning_rate=0.1,
                balanced_weights=False,
                recurrent_dropout=0.2,
                n_workers=5,
                dropout=0.2,
                loss='crossentropy',
                window=None,
                classifier='avgpool',
                random_noise=None,
                steps_per_epoch=20,
                freeze_up_to=None,
                share_feature_layer=False):
    train_multi_task(
    batch_size=batch_size,
    feature_extractor='vgg16',
    epochs=epochs,
    balanced_weights=balanced_weights,
    window=window,
    num_mels=num_mels,
    dropout=dropout,
    tasks=tasks,
    loss=loss,
    classifier=classifier,
    directory=data_path,
    experiment_base_path=experiment_base_path,
    multi_task_setup=multi_task_setup,
    freeze_up_to=freeze_up_to,
    random_noise=random_noise,
    steps_per_epoch=steps_per_epoch,
    optimizer=OPTIMIZERS[optimizer],
    share_feature_layer=share_feature_layer)

    
# @rnn.command(name='single-task')
@click.option('-dp',
              '--data-path',
              required=True,
              help='Directory of data files.',
              type=click.Path(file_okay=False))
@click.option('-t',
              '--task',
              required=True,
              help='Name of the task that is trained.',
              type=str)
@click.option('-tr',
              '--train-csv',
              nargs=1,
              required=True,
              help='Path to training csv file.',
              type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('-v',
              '--val-csv',
              required=True,
              help='Path to validation csv file.',
              nargs=1,
              type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('-te',
              '--test-csv',
              required=True,
              help='Path to validation csv file.',
              nargs=1,
              type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    '-bs',
    '--batch-size',
    type=int,
    help='Define batch size.',
    default=64,
)
@click.option(
    '-nm',
    '--num-mels',
    type=int,
    help='Number of mel bands in spectrogram.',
    default=128,
)
@click.option(
    '-mfcc',
    '--num-mfccs',
    type=int,
    help='Number of mfccs.',
    default=None,
)
@click.option(
    '-e',
    '--epochs',
    type=int,
    help='Define max number of training epochs.',
    default=1000,
)
@click.option(
    '-p',
    '--patience',
    type=int,
    help=
    'Define patience before early stopping / reducing learning rate in epochs.',
    default=20,
)
@click.option(
    '-im',
    '--initial-model',
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help='Initial model for resuming training.',
    default=None,
)
@click.option('-bw',
              '--balanced-weights',
              help='Automatically set balanced class weights.',
              is_flag=True)
@click.option(
    '-lr',
    '--learning-rate',
    type=click.FloatRange(0, 1),
    default=0.1,
    help='Initial earning rate for optimizer.',
)
@click.option(
    '-do',
    '--dropout',
    type=click.FloatRange(0, 1),
    default=0.2,
    help='Dropout for rnn.',
)
@click.option(
    '-ebp',
    '--experiment-base-path',
    type=click.Path(writable=True, readable=True),
    help='Basepath where logs and checkpoints should be stored.',
    default='./residual-adapters',
)
@click.option('-o',
              '--optimizer',
              type=click.Choice(OPTIMIZERS.keys()),
              help='Optimizer used for training.',
              default='sgd')
@click.option('-w',
              '--window',
              type=float,
              help='Window size in seconds.',
              default=None)
@click.option('-rn',
              '--random-noise',
              type=float,
              help='Power of random gaussian noise to add to input spectrograms for augmentation.',
              default=None)
@click.option('-l',
              '--loss',
              type=click.Choice(['crossentropy', 'focal', 'ordinal']),
              help='Classification loss.',
              default='crossentropy')
@click.option('-m',
              '--mode',
              type=click.Choice(['scratch', 'adapters', 'last-layer', 'finetune']))
@click.option('-nw',
              '--n-workers',
              help='Number of training workers.',
              type=int,
              default=5)
@click.option('-dp',
              '--down-pool',
              help='Downpooling factor for input MFCCs.',
              type=int,
              default=8)
@click.option('-hd',
              '--hidden-dim',
              help='Dimension of hidden rnn layers.',
              type=int,
              default=512)
@click.option('-nl',
              '--number-of-layers',
              help='Number of hidden rnn layers.',
              type=int,
              default=2)
@click.option('-ct',
              '--cell-type',
              type=click.Choice(['lstm', 'gru']),
              default='lstm')
@click.option('-ib',
              '--input-bn',
              help='Introduce BatchNormalization directly after input.',
              is_flag=True)
@click.option('-sfl',
              '--share-feature-layer',
              help='Share the feature layer (weighted attention of deep features) between tasks.',
              is_flag=True)
@click.option('-sa',
              '--share-attention',
              help='Share the self attention layers between tasks.',
              is_flag=True)
@click.option('-ua',
              '--use-attention',
              help='Use self attention layers.',
              is_flag=True)
@click.option('-ip',
              '--input-projection',
              help='Use a dense input projection layer.',
              is_flag=True)
@click.option('--bidirectional/--unidirectional', default=False)
def rnn_single_task(data_path,
                task,
                train_csv='train.csv',
                val_csv='val.csv',
                test_csv='test.csv',
                batch_size=64,
                num_mels=128,
                num_mfccs=40,
                number_of_layers=2,
                epochs=1000,
                initial_model=None,
                experiment_base_path='./experiments/residual-adapters',
                optimizer='sgd',
                learning_rate=0.1,
                balanced_weights=False,
                dropout=0.2,
                hidden_dim=512,
                cell_type='lstm',
                bidirectional=False,
                n_workers=5,
                down_pool=8,
                loss='crossentropy',
                window=None,
                patience=20,
                mode='scratch',
                random_noise=None,
                input_bn=False,
                share_feature_layer=False,
                use_attention=True,
                share_attention=False,
                input_projection=True):
    learnall = mode == 'scratch' or mode == 'finetune'
    learnall_classifier = mode == 'scratch' or mode == 'finetune'
    last_layer_only = True if mode == 'last-layer' else False

    train_single_task(initial_weights=initial_model,
                    feature_extractor='rnn',
                     batch_size=batch_size,
                     epochs=epochs,
                     balanced_weights=balanced_weights,
                     window=window,
                     num_mels=num_mels,
                     num_mfccs=num_mfccs,
                     dropout=dropout,
                     task=task,
                     mode=mode,
                     loss=loss,
                     directory=data_path,
                     train_csv=train_csv,
                     number_of_layers=number_of_layers,
                     val_csv=val_csv,
                     test_csv=test_csv,
                     experiment_base_path=experiment_base_path,
                     random_noise=random_noise,
                     learnall=learnall,
                     last_layer_only=last_layer_only,
                     initial_learning_rate=learning_rate,
                     optimizer=OPTIMIZERS[optimizer],
                     patience=patience,
                     hidden_dim=hidden_dim,
                     down_pool=down_pool,
                     cell=cell_type,
                     bidirectional=bidirectional,
                     n_workers=n_workers,
                     input_bn=input_bn,
                     share_feature_layer=share_feature_layer,
                     use_attention=use_attention,
                     share_attention=share_attention,
                     input_projection=input_projection)
    
    
# @rnn.command(name='multi-task')
@click.option('-dp',
              '--data-path',
              required=True,
              help='Directory of data files.',
              type=click.Path(file_okay=False))
@click.option('-mts',
              '--multi-task-setup',
              required=True,
              help='Directory with the setup csvs for each task in a separate folder.',
              type=click.Path(file_okay=False))
@click.option('-t',
              '--tasks',
              required=True,
              help='Names of the tasks that are trained.',
              cls=OptionEatAll)
@click.option(
    '-bs',
    '--batch-size',
    type=int,
    help='Define batch size.',
    default=64,
)
@click.option(
    '-nm',
    '--num-mels',
    type=int,
    help='Number of mel bands in spectrogram.',
    default=128,
)
@click.option(
    '-mfcc',
    '--num-mfccs',
    type=int,
    help='Number of mfccs.',
    default=None,
)
@click.option(
    '-e',
    '--epochs',
    type=int,
    help='Define max number of training epochs.',
    default=50,
)
@click.option(
    '-spe',
    '--steps-per-epoch',
    type=int,
    help=
    'Number of training steps for each artificial epoch.',
    default=20
)
@click.option('-bw',
              '--balanced-weights',
              help='Automatically set balanced class weights.',
              is_flag=True)
@click.option(
    '-lr',
    '--learning-rate',
    type=click.FloatRange(0, 1),
    default=0.1,
    help='Initial earning rate for optimizer.',
)
@click.option(
    '-do',
    '--dropout',
    type=click.FloatRange(0, 1),
    default=0.2,
    help='Dropout for rnn.',
)
@click.option(
    '-ebp',
    '--experiment-base-path',
    type=click.Path(writable=True, readable=True),
    help='Basepath where logs and checkpoints should be stored.',
    default='./residual-adapters',
)
@click.option('-o',
              '--optimizer',
              type=click.Choice(OPTIMIZERS.keys()),
              help='Optimizer used for training.',
              default='sgd')
@click.option('-w',
              '--window',
              type=float,
              help='Window size in seconds.',
              default=None)
@click.option('-rn',
              '--random-noise',
              type=float,
              help='Power of random gaussian noise to add to input spectrograms for augmentation.',
              default=None)
@click.option('-l',
              '--loss',
              type=click.Choice(['crossentropy', 'focal', 'ordinal']),
              help='Classification loss.',
              default='crossentropy')
@click.option('-hd',
              '--hidden-dim',
              help='Dimension of hidden rnn layers.',
              type=int,
              default=512)
@click.option('-dp',
              '--down-pool',
              help='Downpooling factor for input MFCCs.',
              type=int,
              default=8)
@click.option('-nl',
              '--number-of-layers',
              help='Number of hidden rnn layers.',
              type=int,
              default=2)
@click.option('-ct',
              '--cell-type',
              type=click.Choice(['lstm', 'gru']),
              default='lstm')
@click.option('--bidirectional/--unidirectional', default=False)
@click.option('-ib',
              '--input-bn',
              help='Introduce BatchNormalization directly after input.',
              is_flag=True)
@click.option('-sfl',
              '--share-feature-layer',
              help='Share the feature layer (weighted attention of deep features) between tasks.',
              is_flag=True)
@click.option('-sa',
              '--share-attention',
              help='Share the self attention layers between tasks.',
              is_flag=True)
@click.option('-ua',
              '--use-attention',
              help='Use self attention layers.',
              is_flag=True)
@click.option('-ip',
              '--input-projection',
              help='Use a dense input projection layer.',
              is_flag=True)
def rnn_multi_task(data_path,
                tasks,
                multi_task_setup,
                batch_size=64,
                num_mels=128,
                num_mfccs=40,
                number_of_layers=2,
                epochs=1000,
                steps_per_epoch=50,
                experiment_base_path='./experiments/residual-adapters',
                optimizer='sgd',
                learning_rate=0.1,
                balanced_weights=False,
                dropout=0.2,
                hidden_dim=512,
                down_pool=8,
                cell_type='lstm',
                bidirectional=False,
                loss='crossentropy',
                window=None,
                random_noise=None,
                input_bn=False,
                share_feature_layer=False,
                use_attention=True,
                share_attention=False,
                input_projection=False):
    

    train_multi_task(feature_extractor='rnn',
                     multi_task_setup=multi_task_setup,
                     tasks=tasks,
                     steps_per_epoch=steps_per_epoch,
                     batch_size=batch_size,
                     epochs=epochs,
                     balanced_weights=balanced_weights,
                     window=window,
                     num_mels=num_mels,
                     num_mfccs=num_mfccs,
                     dropout=dropout,
                     loss=loss,
                     directory=data_path,
                     number_of_layers=number_of_layers,
                     experiment_base_path=experiment_base_path,
                     random_noise=random_noise,
                     initial_learning_rate=learning_rate,
                     optimizer=OPTIMIZERS[optimizer],
                     hidden_dim=hidden_dim,
                     down_pool=down_pool,
                     cell=cell_type,
                     bidirectional=bidirectional,
                     input_bn=input_bn,
                     share_feature_layer=share_feature_layer,
                     use_attention=use_attention,
                     share_attention=share_attention,
                     input_projection=input_projection)



if __name__ == '__main__':
    cli()