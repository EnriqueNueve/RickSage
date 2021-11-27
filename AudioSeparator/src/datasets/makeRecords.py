"""
makeRecords.py

# Make Fuss train
python3 makeRecords.py --dataset fuss --train True --val False --eval False

# Make Fuss val
python3 makeRecords.py --dataset fuss --train False --val True --eval False

# Make Fuss eval
python3 makeRecords.py --dataset fuss --train False --val False --eval True

# Make Fuss train dm with n_samples
python3 makeRecords.py --dataset fuss_train_dm --n_samples 1000 --train False --val False --eval False

"""

import os
import glob
import argparse
import multiprocessing as mp
from itertools import repeat

import random
import numpy as np
import tensorflow as tf

import librosa
#import sounddevice as sd


#########################################
#///////////////////////////////////////#
#########################################


AUTOTUNE = tf.data.experimental.AUTOTUNE
SAMPLERATE_HZ = 8000


#########################################
#///////////////////////////////////////#
#########################################


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def to_tfrecord(mix, source):
    n_source = np.array(source.shape[0])
    feature = {
        'mix': _bytes_feature(mix.tobytes()),
        'source': _bytes_feature(source.tobytes()),
        'n_source': _int64_feature(n_source)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def to_tfrecord_dm(background_samples,foreground_samples):
    n_source = int(np.array(foreground_samples.shape[0]))
    feature = {
        'background': _bytes_feature(background_samples.tobytes()),
        'foreground': _bytes_feature(foreground_samples.tobytes()),
        'n_source': _int64_feature(n_source)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def chunks(l, n):
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode="constant", constant_values=0)


#########################################
#///            FUSS                 ///#
#########################################


def writeRecordFuss(source_file_paths,index,tf_path):
    with tf.io.TFRecordWriter(tf_path+'/'+str(index)+'.tfrecord') as out:
        for folder_path in source_file_paths:
            mix_wav_path = folder_path.split("_")[0]+".wav"
            source_file_paths = glob.glob(folder_path+'/*.wav')
            source_file_paths.sort()

            # Load mixed sample
            mix_audio, _ = librosa.load(mix_wav_path, sr=SAMPLERATE_HZ)
            mix_audio = mix_audio[:79872]

            # Load source samples
            source_sounds = []
            for source_path in source_file_paths:
                source_audio, _ = librosa.load(source_path, sr=SAMPLERATE_HZ)
                source_sounds.append(source_audio[:79872])

            # Add dummy source samples if needed
            if len(source_sounds) != 4:
                for i in range(4-len(source_sounds)):
                    dummy_source = np.zeros((79872,),dtype='float32')
                    source_sounds.append(dummy_source)
            source = np.stack(source_sounds)

            # Encode [mix, source] pair to TFRecord format
            example = to_tfrecord(mix_audio, source)
            # Write serialized example to TFRecord file
            out.write(example.SerializeToString())

def writeRecordFussSudoEfficientDM(background_paths, foreground_paths, index, tf_path):
    with tf.io.TFRecordWriter(tf_path+'/'+str(index)+'.tfrecord') as out:
        for i, background_path in enumerate(background_paths):
            # Load data and apply time speed augment between .95 and 1.05
            foreground_sample_paths = random.sample(foreground_paths, 3)

            background_sample, _ = librosa.load(background_path, sr=8000)
            background_sample = librosa.effects.time_stretch(background_sample,random.uniform(.95,1.05))
            if len(background_sample) < 79872: #pad
                background_sample = pad_along_axis(background_sample,79872)
            elif len(background_sample) > 79872: # cut
                background_sample = background_sample[:79872]

            foreground_sample_1, _ = librosa.load(foreground_sample_paths[0], sr=8000)
            foreground_sample_1 = librosa.effects.time_stretch(foreground_sample_1,random.uniform(.95,1.05))
            if len(foreground_sample_1) < 79872: #pad
                foreground_sample_1 = pad_along_axis(foreground_sample_1,79872)
            elif len(foreground_sample_1) > 79872: # cut
                foreground_sample_1 = foreground_sample_1[:79872]

            foreground_sample_2, _ = librosa.load(foreground_sample_paths[1], sr=8000)
            foreground_sample_2 = librosa.effects.time_stretch(foreground_sample_2,random.uniform(.95,1.05))
            if len(foreground_sample_2) < 79872: #pad
                foreground_sample_2 = pad_along_axis(foreground_sample_2,79872)
            elif len(foreground_sample_2) > 79872: # cut
                foreground_sample_2 = foreground_sample_2[:79872]

            foreground_sample_3, _ = librosa.load(foreground_sample_paths[2], sr=8000)
            foreground_sample_3 = librosa.effects.time_stretch(foreground_sample_3,random.uniform(.95,1.05))
            if len(foreground_sample_3) < 79872: #pad
                foreground_sample_3 = pad_along_axis(foreground_sample_3,79872)
            elif len(foreground_sample_3) > 79872: # cut
                foreground_sample_3 = foreground_sample_3[:79872]

            # Shape data
            foreground_samples = np.stack([foreground_sample_1,foreground_sample_2,foreground_sample_3])

            # Encode [source] to TFRecord format
            example = to_tfrecord_dm(background_sample,foreground_samples)
            # Write serialized example to TFRecord file
            out.write(example.SerializeToString())

def buildFussTrainDM(configs):
    fuss_path = 'fuss'
    fuss_train = fuss_path+'/train'

    if not os.path.exists(fuss_path+'/tf_train_dm'):
        os.makedirs(fuss_path+'/tf_train_dm')

    train_source_file_paths = glob.glob(fuss_train+"/*/")

    all_files = []
    for dir_path in train_source_file_paths:
        for file in os.listdir(dir_path):
            all_files.append(dir_path+file)

    background_paths = [k for k in all_files if 'background' in k]
    foreground_paths = [k for k in all_files if 'foreground' in k]

    background_paths.sort()
    foreground_paths.sort()

    n_samples = configs.n_samples
    train_dm_index = list(range(n_samples//100))
    background_path_batches, foreground_path_batches = [], []
    for i in range(n_samples//100):
        background_path_batches.append(random.sample(background_paths,100))
        foreground_path_batches.append(random.sample(foreground_paths,100))

    # Make tf Record for train sudo efficient DM
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(writeRecordFussSudoEfficientDM, zip(background_path_batches,\
                    foreground_path_batches,train_dm_index,repeat(fuss_path+'/tf_train_dm')))

def buildFuss(configs):
    fuss_path = 'fuss'
    fuss_eval = fuss_path+'/eval'
    fuss_train = fuss_path+'/train'
    fuss_validation = fuss_path+'/validation'

    eval_source_file_paths = glob.glob(fuss_eval+"/*/")
    eval_source_file_paths.sort()

    eval_mix_audio_paths = glob.glob(fuss_eval+"/*.wav")
    eval_mix_audio_paths.sort()

    train_source_file_paths = glob.glob(fuss_train+"/*/")
    train_source_file_paths.sort()

    train_mix_audio_paths = glob.glob(fuss_train+"/*.wav")
    train_mix_audio_paths.sort()

    validation_source_file_paths = glob.glob(fuss_validation+"/*/")
    validation_source_file_paths.sort()

    validation_mix_audio_paths = glob.glob(fuss_validation+"/*.wav")
    validation_mix_audio_paths.sort()

    print("Number of train sample: {}".format(len(train_mix_audio_paths)))
    print("Number of val sample: {}".format(len(validation_mix_audio_paths)))
    print("Number of eval sample: {}".format(len(eval_mix_audio_paths)))

    if configs.train == True:
        if not os.path.exists(fuss_path+'/tf_train'):
            os.makedirs(fuss_path+'/tf_train')

        train_source_file_paths_cuts = chunks(train_source_file_paths,100)
        train_index = list(range(len(train_source_file_paths_cuts)))

        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.starmap(writeRecordFuss, zip(train_source_file_paths_cuts,train_index,repeat(fuss_path+'/tf_train')))

    if configs.val == True:
        if not os.path.exists(fuss_path+'/tf_val'):
            os.makedirs(fuss_path+'/tf_val')

        validation_source_file_paths_cuts = chunks(validation_source_file_paths,100)
        val_index = list(range(len(validation_source_file_paths_cuts)))

        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.starmap(writeRecordFuss, zip(validation_source_file_paths_cuts,val_index,repeat(fuss_path+'/tf_val')))

    if configs.eval == True:
        if not os.path.exists(fuss_path+'/tf_eval'):
            os.makedirs(fuss_path+'/tf_eval')

        eval_source_file_paths_cuts = chunks(eval_source_file_paths,100)
        eval_index = list(range(len(eval_source_file_paths_cuts)))

        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.starmap(writeRecordFuss, zip(eval_source_file_paths_cuts,eval_index,repeat(fuss_path+'/tf_eval')))


#########################################
#///////////////////////////////////////#
#########################################


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_configs():
    parser = argparse.ArgumentParser(
        description="Build tfRecord files"
    )
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--train', type=str2bool, required=True, default=False)
    parser.add_argument('--eval', type=str2bool, required=True, default=False)
    parser.add_argument('--val', type=str2bool, required=True, default=False)
    parser.add_argument('--n_samples', type=int, required=False, default=False)
    configs = parser.parse_args()

    return configs


#########################################
#///////////////////////////////////////#
#########################################


def main():
    configs = get_configs()

    if configs.dataset == 'fuss':
        buildFuss(configs)
    elif configs.dataset == 'fuss_train_dm':
        if isinstance(configs.n_samples, int) == False or configs.n_samples < 0:
            raise ValueError('Improper value for n_samples was passed for fuss_train_dm')
        buildFussTrainDM(configs)

#########################################
#///////////////////////////////////////#
#########################################


if __name__ == "__main__":
    main()
