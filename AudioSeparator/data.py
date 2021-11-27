import librosa
import numpy as np
import tensorflow as tf
import glob


#########################################
#///////////////////////////////////////#
#########################################


def tf_shuffle_axis(value, rank, axis=0, seed=None, name=None):
    perm = list(range(rank))
    perm[axis], perm[0] = perm[0], perm[axis]
    value = tf.random.shuffle(tf.transpose(value, perm=perm))
    value = tf.transpose(value, perm=perm)
    return value

def parse_tfrecord_fuss_dm(example,batch_size):
    feature_description = {
        'background': tf.io.FixedLenFeature([], tf.string),
        'foreground': tf.io.FixedLenFeature([], tf.string),
        'n_source': tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_example(example, feature_description)

    background = tf.io.decode_raw(
        example['background'], out_type='float32', little_endian=True, fixed_length=None, name=None
    )

    foreground = tf.io.decode_raw(
        example['foreground'], out_type='float32', little_endian=True, fixed_length=None, name=None
    )

    # Appply Dynamix mixing
    foreground = tf.reshape(foreground,(batch_size*3,-1))
    foreground_mask = tf.transpose(tf.random.categorical(tf.math.log([[0.5, 0.5]]),batch_size*3))
    foreground_mask = tf.cast(foreground_mask,dtype='float32')
    foreground *= foreground_mask
    foreground = tf.reshape(foreground,(batch_size,3,-1))

    background = tf.reshape(background,(batch_size,1,-1))

    # Shuffle foreground and background on each sample within batch
    source = tf.concat([foreground,background], axis=1)
    source = tf_shuffle_axis(source,3,axis=1)

    # Apply gain
    gain = tf.random.uniform((batch_size,4,1),-1,1)
    source *= gain

    # Form mix and scale
    mixture = tf.math.reduce_sum(source,axis=1)
    max_amp = tf.math.reduce_max(mixture)
    mix_scaling = 1/max_amp*.9
    source = mix_scaling*source
    mixture = mix_scaling*mixture
    return tf.clip_by_value(mixture,-1,1), tf.clip_by_value(source,-1,1)

def parse_tfrecord_fuss(example,batch_size):
    feature_description = {
        'mix': tf.io.FixedLenFeature([], tf.string),
        'source': tf.io.FixedLenFeature([], tf.string),
        'n_source': tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_example(example, feature_description)

    mix = tf.io.decode_raw(
        example['mix'], out_type='float32', little_endian=True, fixed_length=None, name=None
    )

    source = tf.io.decode_raw(
        example['source'], out_type='float32', little_endian=True, fixed_length=None, name=None
    )

    source = tf.reshape(source,(batch_size,4,-1))
    source = tf_shuffle_axis(source,3,axis=1)

    return (mix, source)


#########################################
#///////////////////////////////////////#
#########################################


def parse_tfrecord(example,n_source,batch_size):
    feature_description = {
        'mix': tf.io.FixedLenFeature([], tf.string),
        'source': tf.io.FixedLenFeature([], tf.string),
        'n_source': tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_example(example, feature_description)

    mix = tf.io.decode_raw(
        example['mix'], out_type='float32', little_endian=True, fixed_length=None, name=None
    )

    source = tf.io.decode_raw(
        example['source'], out_type='float32', little_endian=True, fixed_length=None, name=None
    )

    source = tf.reshape(source,(batch_size,n_source,-1))

    return (mix, source)


#########################################
#///////////////////////////////////////#
#########################################


def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode="constant", constant_values=0)


#########################################
#///////////////////////////////////////#
#########################################


def getData(configs, AUTO, variant):
    if configs['data_set'] == "/datasets/sample_data":
        mix_data, target_data = getSampleData(configs)
        return mix_data, target_data
    elif configs['data_set'] == "/datasets/fuss":
        batch_size = configs['batch_size']
        if configs['gpu'] > 1:
            batch_size=batch_size*configs['gpu']
        dataset = getFussData(AUTO,batch_size,variant,configs['dm'])
        return dataset


#########################################
#///////////////////////////////////////#
#########################################


def getFussData(AUTO,batch_size=2,variant='eval',dm=False):
    """ Loads tfRecord files
    Parameters
    ----------
    record_files : str or list of str
        paths of tfRecord files
    Returns
    -------
    tf.data.Dataset
        data loader for keras model
    """

    if variant == 'train' and dm == True:
        print("made is to dynamic train for fuss")
        data_path = 'datasets/fuss/tf_train_dm'
        record_files = glob.glob(data_path+"/*.tfrecord")
        print(record_files)

        # Disregard data order in favor of reading speed
        files_ds = tf.data.Dataset.list_files(record_files)
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        files_ds = files_ds.with_options(ignore_order)

        dataset = tf.data.TFRecordDataset(files_ds, num_parallel_reads=AUTO)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(lambda x: parse_tfrecord_fuss_dm(x,batch_size), num_parallel_calls=AUTO)
        return dataset.prefetch(buffer_size=AUTO)

    if variant == 'train':
        data_path = 'datasets/fuss/tf_train'
    elif variant == 'val':
        data_path = 'datasets/fuss/tf_val'
    elif variant == 'eval':
        data_path = 'datasets/fuss/tf_eval'

    record_files = glob.glob(data_path+"/*.tfrecord")

    # Disregard data order in favor of reading speed
    files_ds = tf.data.Dataset.list_files(record_files)
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    files_ds = files_ds.with_options(ignore_order)

    dataset = tf.data.TFRecordDataset(files_ds, num_parallel_reads=AUTO)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x: parse_tfrecord_fuss(x,batch_size), num_parallel_calls=AUTO)
    return dataset.prefetch(buffer_size=AUTO)


#########################################
#///////////////////////////////////////#
#########################################


def getSampleData(configs):
    num_chunks = (
        configs['samplerate_hz']
        * configs['max_input_length_in_seconds']
        // configs['network_chunk_size']
    )
    model_signal_length = configs['network_chunk_size'] * num_chunks

    music_data, music_fs = librosa.load(
        "data/sample_data/street_music_sample.wav", sr=configs['samplerate_hz']
    )
    dog_data, dog_fs = librosa.load(
        "data/sample_data/dog_sample.wav", sr=configs['samplerate_hz']
    )

    # Increase dog bark
    dog_data = dog_data * 2

    if librosa.get_duration(y=music_data, sr=music_fs) > model_signal_length:
        music_data = music_data[:model_signal_length]
    elif librosa.get_duration(y=music_data, sr=music_fs) < model_signal_length:
        music_data = pad_along_axis(music_data, model_signal_length)

    if librosa.get_duration(y=dog_data, sr=music_fs) > model_signal_length:
        dog_data = dog_data[:model_signal_length]
    elif librosa.get_duration(y=dog_data, sr=dog_fs) < model_signal_length:
        dog_data = pad_along_axis(dog_data, model_signal_length)

    # Mix audio
    mix_data = music_data + dog_data
    mix_data = mix_data[np.newaxis, ...]

    # Stack audio for target
    target = np.stack([dog_data, music_data])
    target_data = target[np.newaxis, ...]

    return mix_data, target_data
