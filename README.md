# EmoNet

This repository contains the code for the EmoNet framework for multi-domain learning networks for audio tasks.

## Installation

All dependencies can be installed via pip from the requirements.txt:

```bash
pip install -r requirements.txt
```

It is advisable to do this from within a newly created virtual environment.

## Usage

The basic commandline is accessible from the repository's basedirectory by calling:

```bash
python -m emo-net.cli --help
```

This prints a help message specifying the list of subcommands. For each subcommand, more help is available via:

```bash
python -m emo-net.cli [subcommand] --help
```

### Data Preparation

The toolkit can be used for arbitrary audio classification tasks. To prepare your dataset, resample all audio content to 16kHz wav files (e.g. with ffmpeg). Afterwards, you need label files in .csv format that specify the categorical target for each sample in the training, development and test partitions, i.e., three files "train.csv", "devel.csv" and "test.csv". The files must include the path to each audio file in the first column - relative to a common basedirectory - and a categorical label in the second column. A header line "file,label" should be included.

### Command line options

The CLI has a nested structure, i.e., it uses two layers of subcommands. The first subcommand specifies the type of neural network architecture that is used. Here, "cnn" gives access to the ResNet architecture which also includes residual adapters, based on the training setting. Two other options, "rnn" and "fusion" are also included but untested and in early stages of development. The rest of this guide will therefore focus on the "cnn" subcommand. After specifying the model type, two distinct subcommands are accessible: "single-task" and "multi-task", which refer to the type of training procedure. For single task, training is performed on one database at a time specified by its basedirectory and the label files for train, validation and developments:

```bash
python -m emo-net.cli -v cnn single-task -t [taskName] --data-path /path/to/task/wavs -tr train.csv -v devel.csv -te test.csv
```

One additional parameter is needed that defines the type of training performed. Here, the choice can be made between tuning a fresh model from scratch (`-m scratch`), fully finetuning an existing model (`-m finetune`), training only the classifier head (`-m last-layer`) and the residual adapter approach (`-m adapters`). For the last three methods, a pre-trained model has to be loaded by specifying the path to its weights via `-im /path/to/weights.h5`. While all other parameters have sensible default values, the full list is given below:

| Option                          | Type                                      | Description                                                                                                                                                   |
| ------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| -dp, --data-path                | DIRECTORY                                 | Directory of data files. [required]                                                                                                                           |
| -t, --task                      | TEXT                                      | Name of the task that is trained. [required]                                                                                                                  |
| -tr, --train-csv                | FILE                                      | Path to training csv file. [required]                                                                                                                         |
| -v, --val-csv                   | FILE                                      | Path to validation csv file. [required]                                                                                                                       |
| -te, --test-csv                 | FILE                                      | Path to test csv file. [required]                                                                                                                             |
| -bs, --batch-size               | INTEGER                                   | Define batch size.                                                                                                                                            |
| -nm, --num-mels                 | INTEGER                                   | Number of mel bands in spectrogram.                                                                                                                           |
| -e, --epochs                    | INTEGER                                   | Define max number of training epochs.                                                                                                                         |
| -p, --patience                  | INTEGER                                   | Define patience before early stopping / reducing learning rate in epochs.                                                                                     |
| -im, --initial-model            | FILE                                      | Initial model for resuming training.                                                                                                                          |
| -bw, --balanced-weights         | FLAG                                      | Automatically set balanced class weights.                                                                                                                     |
| -lr, --learning-rate            | FLOAT                                     | Initial earning rate for optimizer.                                                                                                                           |
| -do, --dropout                  | FLOAT                                     | Dropout for the two positions (after first and second convolution of each block).                                                                             |
| -ebp, --experiment-base-path    | PATH                                      | Basepath where logs and checkpoints should be stored.                                                                                                         |
| -o, --optimizer                 | [sgd\|rmsprop\|adam\|adadelta]            | Optimizer used for training.                                                                                                                                  |
| -N, --number-of-resnet-blocks   | INTEGER                                   | Number of convolutional blocks in the ResNet layers.                                                                                                          |
| -nf, --number-of-filters        | INTEGER                                   | Number of filters in first convolutional block.                                                                                                               |
| -wf, --widen-factor             | INTEGER                                   | Widen factor of wide ResNet                                                                                                                                   |
| -c, --classifier                | [avgpool\|FCNAttention]                   | The classification top of the network architeture. Choose between simple pooling + dense layer (needs fixed window size) and fully convolutional attention.   |
| -w, --window                    | FLOAT                                     | Window size in seconds.                                                                                                                                       |
| -l, --loss                      | [crossentropy\|focal\|ordinal]            | Classification loss. Ordinal loss ues sorted class labels.                                                                                                    |
| -m, --mode                      | [scratch\|adapters\|last-layer\|finetune] | Type of training to be performed.                                                                                                                             |
| -sfl, --share-feature-layer     | FLAG                                      | Share the feature layer (weighted attention of deep features) between tasks.                                                                                  |
| -iwd, --individual-weight-decay | FLAG                                      | Set weight decay in adapters according to size of training dataset. Smaller datasets will have larger weight decay to keep closer to the pre-trained network. |
| --help                          | FLAG                                      | Show this message and exit.                                                                                                                                   |

The "multi-task" command line slightly differs from the one described above. The most notable difference is in how the data is passed. Instead of passing individual .csv files for each partition, a directory - "--multi-task-setup" - which contains a folder with "train.csv", "val.csv" and "test.csv" files for each database has to be specified. Additionally, "-t" now is used to specify a list of databases (subfolders of the multi task setup) that should be used for training. As multi-domain training is done in a round-robin fashion, there is no predefined notion of a training epoch. Therefore, an additional option ("--steps-per-epoch") is used to define the size of an artificial training epoch. These additional parameters are also given in the table below.

| Option                   | Type      | Description                                                                                                       |
| ------------------------ | --------- | ----------------------------------------------------------------------------------------------------------------- |
| -dp, --data-path         | DIRECTORY | Directory of wav files. [required]                                                                                |
| -mts, --multi-task-setup | DIRECTORY | Directory with the setup csvs ("train.csv", "val.csv", "test.csv") for each task in a separate folder. [required] |
| -t, --tasks              | TEXT      | Names of the tasks that are trained. [required]                                                                   |
| -spe, --steps-per-epoch  | INTEGER   | Number of training steps for each artificial epoch.                                                               |
