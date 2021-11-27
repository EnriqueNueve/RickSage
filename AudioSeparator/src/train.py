"""
train.py

Examples:

# Default call
python train.py

# Call with type of dataset
python3 train.py --config configs/rha_train_default.yaml

python3 train.py --config configs/rha_train_dm_default.yaml


"""

"""
TO DO
* Make two training routines: one for gpu where you can select #-gpu and one for cpu
    ~ Look into tf Data distribute
"""

import argparse, yaml
import os
from argparse import Namespace
from model import *
from data import *
from utils import *


#########################################
#///////////////////////////////////////#
#########################################


AUTOTUNE = tf.data.experimental.AUTOTUNE


#########################################
#///////////////////////////////////////#
#########################################


def get_configs(yaml_test_overide=None):
    parser = argparse.ArgumentParser(
        description="Training routine for Cocktails with Robots"
    )
    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='configs/default.yaml')
    parser.add_argument('--args', help="use yaml file or not", type=bool, required=False, default=False)
    parser.add_argument('--train_pickup', help="load or not load weights to continue training", type=bool, required=False, default=False)

    # train
    parser.add_argument("--gpu", default="0", type=int, help="GPU num for training")
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=1, type=int)

    # model
    parser.add_argument("--model_type", type=str, default="RHA")
    parser.add_argument("--network_n_sources", default=2, type=int, help="Number of sources being seperated")
    parser.add_argument("--network_num_filters_in_encoder", default=64, type=int, help="Number of filters in encoder")
    parser.add_argument("--network_encoder_filter_length", default=2, type=int, help="Length of encoder filter")
    parser.add_argument("--network_num_head_per_att", default=8, type=int, help="Number of heads in each MHA")
    parser.add_argument("--network_dim_key_att", default=1024, type=int, help="Dim of key in attention")
    parser.add_argument("--network_num_tran_blocks", default=1, type=int, help="Number of Transformer Blocks")
    parser.add_argument("--network_num_chop_blocks", default=1, type=int, help="Number of Chop Blocks")
    parser.add_argument("--network_chunk_size", default=256, type=int, help="Size of chunk window")

    # dataset
    parser.add_argument("--data_set", default="/data/sample_data", type=str, help="Dataset to train on")
    parser.add_argument('--dm', help="use dynamic mixing or not", type=bool, required=False, default=False)
    parser.add_argument("--max_input_length_in_seconds", default=5, type=int, help="Max length in seconds of audio clip")
    parser.add_argument("--samplerate_hz", default=8000, type=int, help="Sample rate of audio in Hz")

    # optimizer
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate for training")

    configs = parser.parse_args()


    if yaml_test_overide != None: # Used for unit test
        opt = vars(configs)
        with open(yaml_test_overide) as file:
            args = yaml.load(file, Loader=yaml.FullLoader)
        opt.update(args)
        configs = opt
        return configs

    if not configs.args:  # args priority is higher than yaml
        opt = vars(configs)
        with open(configs.config) as file:
            args = yaml.load(file, Loader=yaml.FullLoader)
        opt.update(args)
        configs = opt
    else:  # yaml priority is higher than args
        with open(configs['config']) as file:
            opt = yaml.load(file, Loader=yaml.FullLoader)
        opt.update(vars(configs))
        configs = opt
    return configs


#########################################
#///////////////////////////////////////#
#########################################


def trainFuss(configs):
    EXPERIMENT_NAME = configs['config'].split('/')[-1].split('.')[0]

    if checkExperimentBuilt(EXPERIMENT_NAME) == False:
        print('The passed experiment does not exist, making new experiment: {}'.format(EXPERIMENT_NAME))
        buildExperiment(EXPERIMENT_NAME,configs)

    if configs['gpu'] == 0:
        # Get data
        train_dataset = getData(configs, AUTOTUNE, 'train')
        val_dataset = getData(configs, AUTOTUNE, 'val')
        test_dataset = getData(configs, AUTOTUNE, 'eval')

        # Get model
        model = getModel(configs)

        # Callbacks
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath='model_weights/'+EXPERIMENT_NAME+'/'+EXPERIMENT_NAME+'_val_best.ckpt',
                        save_weights_only=True,
                        monitor="val_snr_vary_n_source",
                        mode="max",
                        save_best_only=True,
                    )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_snr_vary_n_source', mode='max' ,factor=0.2,
                                      patience=5, min_lr=0.0001)

        callbacks = [model_checkpoint_callback,reduce_lr]

        # Load weights
        if configs['train_pickup'] == True:
            #model.model.load_weights('model_weights/'+EXPERIMENT_NAME+'/'+EXPERIMENT_NAME+'_val_best.ckpt')
            model.load_weights('model_weights/'+EXPERIMENT_NAME+'/'+EXPERIMENT_NAME+'_val_best.ckpt')
            print('loaded val best weights!')

        # Train model
        history = model.fit(train_dataset.take(1), validation_data=val_dataset.take(1), epochs=configs['epochs'],\
                                      batch_size=configs['batch_size'],callbacks=callbacks)

        # Update train log
        updateTrainLog(EXPERIMENT_NAME,configs,history)

        # Evalute and update result log
        for sample in test_dataset.take(1):
            mix_data, source = sample

        results = model.test_step((mix_data,source))
        updateResultLog(EXPERIMENT_NAME,configs,results)

    elif configs['gpu'] > 1:
        train_dataset = getData(configs, AUTOTUNE, 'train')
        val_dataset = getData(configs, AUTOTUNE, 'val')
        test_dataset = getData(configs, AUTOTUNE, 'eval')


        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        batch_size = configs['batch_size']*strategy.num_replicas_in_sync

        train_dataset  = mirrored_strategy.experimental_distribute_dataset(train_dataset )
        val_dataset = mirrored_strategy.experimental_distribute_dataset(val_dataset)
        test_dataset = mirrored_strategy.experimental_distribute_dataset(test_dataset)

        # Open a strategy scope.
        with strategy.scope():
          # Everything that creates variables should be under the strategy scope.
          # In general this is only model construction & `compile()`.
          model = getModel(configs)

        # Callbacks
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath='model_weights/'+EXPERIMENT_NAME+'/'+EXPERIMENT_NAME+'_val_best.ckpt',
                        save_weights_only=True,
                        monitor="val_snr_vary_n_source",
                        mode="max",
                        save_best_only=True,
                    )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_snr_vary_n_source', mode='max' ,factor=0.2,
                                      patience=5, min_lr=0.0001)

        callbacks = [model_checkpoint_callback,reduce_lr]

        # Load weights
        if configs['train_pickup'] == True:
            #model.model.load_weights('model_weights/'+EXPERIMENT_NAME+'/'+EXPERIMENT_NAME+'_val_best.ckpt')
            model.load_weights('model_weights/'+EXPERIMENT_NAME+'/'+EXPERIMENT_NAME+'_val_best.ckpt')
            print('loaded val best weights!')

        # Train model
        history = model.fit(mix_data,source, validation_data=(mix_data,source), epochs=configs['epochs'],\
                                      batch_size=batch_size,callbacks=callbacks)

        # Update train log
        updateTrainLog(EXPERIMENT_NAME,configs,history)

        # Evalute and update result log
        results = model.test_step(test_dataset)
        updateResultLog(EXPERIMENT_NAME,configs,results)

    # Save weights
    # HAVE SAVE NOT AS VAL BEST BUT FINAL ...
    #model.model.save_weights(configs['train_weight_path'])

    #pass


def trainSampleData(configs):
    pass


#########################################
#///////////////////////////////////////#
#########################################


def main():
    configs = get_configs()

    if configs['data_set'] == '/datasets/fuss':
        trainFuss(configs)
    elif configs['data_set'] == '/datasets/sample_data':
        trainSampleData(configs)


#########################################
#///////////////////////////////////////#
#########################################


if __name__ == "__main__":
    main()
