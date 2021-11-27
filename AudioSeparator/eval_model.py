"""
eval_model.py


Purpose: Used to evaluate final trained model to measure under, equal,
or over source estimation and si-snri for each number of sources.

Examples:

python3 eval_model.py  --config configs/rha_train_dm_default.yaml
"""

import argparse, yaml
import os
from argparse import Namespace
from evaluation import *
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


def evalFuss(configs):
    configs['batch_size'] = 1
    EXPERIMENT_NAME = configs['config'].split('/')[-1].split('.')[0]

    # Check experiment has been ran
    assert checkExperimentBuilt(EXPERIMENT_NAME), "Experiment has not been ran yet, can't evaluate until model is trained!"

    # Load data
    test_dataset = getData(configs, AUTOTUNE, 'eval')

    # Get model
    model = getModel(configs)

    getFinalMetricsFuss(model,test_dataset,EXPERIMENT_NAME,n_samples=5)



#########################################
#///////////////////////////////////////#
#########################################


def main():
    configs = get_configs()

    if configs['data_set'] == '/datasets/fuss':
        evalFuss(configs)


#########################################
#///////////////////////////////////////#
#########################################


if __name__ == "__main__":
    main()
