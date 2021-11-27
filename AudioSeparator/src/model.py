from loss import *
from layers import *

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons.optimizers import AdamW


#########################################
#///////////////////////////////////////#
#########################################


def getSepFormer(configs,print_model):
    return "SepFormer"


#########################################
#///////////////////////////////////////#
#########################################


class RHA(keras.Model):
    def __init__(
        self,
        batch_size,
        model_weights_file,
        num_filters_in_encoder,
        encoder_filter_length,
        chunk_size,
        num_full_chunks,
        num_chop_blocks,
        num_tran_blocks,
        num_head_per_att,
        dim_key_att,
        max_input_length_in_seconds,
        samplerate_hz,
        num_speakers,
    ):
        super(RHA, self).__init__()

        self.num_chop_blocks = num_chop_blocks
        self.batch_size = batch_size
        self.model_weights_file = model_weights_file
        self.max_input_length_in_seconds = max_input_length_in_seconds
        self.num_tran_blocks = num_tran_blocks
        self.num_head_per_att = num_head_per_att
        self.dim_key_att = dim_key_att
        self.encoder_filter_length = encoder_filter_length
        self.num_filters_in_encoder = num_filters_in_encoder
        self.encoder_hop_size = encoder_filter_length // 2
        self.num_full_chunks = num_full_chunks
        self.signal_length_samples = chunk_size * num_full_chunks
        self.chunk_size = chunk_size
        self.chunk_advance = chunk_size // 2
        self.num_overlapping_chunks = num_full_chunks * 2 - 1
        self.num_speakers = num_speakers
        self.samplerate_hz = samplerate_hz
        self.train_clip = 30

        # Build model
        self.model = self.getModel()
        self.loss_tracker = keras.metrics.Mean(name="snr_vary_n_source")
        self.sdri_tracker = keras.metrics.Mean(name="sdri")
        self.sisnri_tracker = keras.metrics.Mean(name="sisnri")

    def getModel(self):
        # Model input
        inputs = tf.keras.Input(self.signal_length_samples)

        # Encoder Block
        z, encoder_out = Encoder(
            num_filters_in_encoder=self.num_filters_in_encoder,
            encoder_filter_length=self.encoder_filter_length,
            samplerate_hz=self.samplerate_hz,
            max_input_length_in_seconds=self.max_input_length_in_seconds,
            chunk_length=self.chunk_size,
            batch_size=self.batch_size,
        )(inputs)


        # Chop Shop
        for i in range(self.num_chop_blocks):
            # Intra ~ Short
            z = tf.reshape(z, (self.batch_size, -1, self.chunk_size))
            _, _, dim_check = z.get_shape()
            if dim_check % self.num_head_per_att != 0:
                for i in range(self.num_tran_blocks):
                    if i > 0:
                        z = tf.reshape(z, (self.batch_size, -1, self.chunk_size))
                    pad_len = self.num_head_per_att - (
                        dim_check % self.num_head_per_att
                    )
                    zero_pad = tf.zeros(
                        (
                            self.batch_size,
                            self.num_overlapping_chunks * self.num_filters_in_encoder,
                            pad_len,
                        )
                    )
                    z_pad = tf.concat([z, zero_pad], axis=-1)
                    x = LinearSineSPETransformerBlock(
                        self.chunk_size + pad_len,
                        self.num_head_per_att,
                        self.dim_key_att,
                    )(z_pad)
                    x = x[:, :, :-pad_len]
                    z = x + z
                    z = tf.reshape(
                        z,
                        (
                            self.batch_size,
                            self.num_overlapping_chunks,
                            self.chunk_size,
                            self.num_filters_in_encoder,
                        ),
                    )
            else:
                for i in range(self.num_tran_blocks):
                    if i > 0:
                        z = tf.reshape(z, (self.batch_size, -1, self.chunk_size))
                    x = LinearSineSPETransformerBlock(
                        self.chunk_size, self.num_head_per_att, self.dim_key_att
                    )(z)
                    z = x + z
                    z = tf.reshape(
                        z,
                        (
                            self.batch_size,
                            self.num_overlapping_chunks,
                            self.chunk_size,
                            self.num_filters_in_encoder,
                        ),
                    )

            # Inter ~ Long
            z = tf.reshape(z, (self.batch_size, -1, self.num_overlapping_chunks))
            _, _, dim_check = z.get_shape()
            if dim_check % self.num_head_per_att != 0:
                # Case 1: padding needed
                for i in range(self.num_tran_blocks):
                    if i > 0:
                        z = tf.reshape(
                            z, (self.batch_size, -1, self.num_overlapping_chunks)
                        )
                    pad_len = self.num_head_per_att - (
                        dim_check % self.num_head_per_att
                    )
                    zero_pad = tf.zeros(
                        (
                            self.batch_size,
                            self.chunk_size * self.num_filters_in_encoder,
                            pad_len,
                        )
                    )
                    z_pad = tf.concat([z, zero_pad], axis=-1)
                    x = LinearSineSPETransformerBlock(
                        self.num_overlapping_chunks + pad_len,
                        self.num_head_per_att,
                        self.dim_key_att,
                    )(z_pad)
                    x = x[:, :, :-pad_len]
                    z = x + z
                z = tf.reshape(
                    z,
                    (
                        self.batch_size,
                        self.num_overlapping_chunks,
                        self.chunk_size,
                        self.num_filters_in_encoder,
                    ),
                )
            else:
                # Case 2: padding not needed
                for i in range(self.num_tran_blocks):
                    if i > 0:
                        z = tf.reshape(z, (self.batch_size, -1, self.chunk_size))
                    x = LinearSineSPETransformerBlock(
                        self.num_overlapping_chunks,
                        self.num_head_per_att,
                        self.dim_key_att,
                    )(z)
                    z = x + z
                    z = tf.reshape(
                        z,
                        (
                            self.batch_size,
                            self.num_overlapping_chunks,
                            self.chunk_size,
                            self.num_filters_in_encoder,
                        ),
                    )
        x = z

        # Decoder Block
        decoded = Decoder(
            signal_length_samples=self.signal_length_samples,
            n_sources=self.num_speakers,
            num_filters_in_encoder=self.num_filters_in_encoder,
            encoder_filter_length=self.encoder_filter_length,
            samplerate_hz=self.samplerate_hz,
            max_input_length_in_seconds=self.max_input_length_in_seconds,
            chunk_length=self.chunk_size,
            batch_size=self.batch_size,
        )(x, encoder_out)

        # Final model
        model = tf.keras.Model(inputs, decoded)
        return model

    def call(self, inputs):
        yh = self.model(inputs)
        return yh

    def train_step(self, inputs):
        X, y = inputs
        y_mix = tf.math.reduce_sum(y, axis=1, keepdims=True)
        with tf.GradientTape() as tape:
            print(X.shape)
            yh = self.model(X)
            print(yh.shape)
            yh = enforce_mixture_consistency_time_domain(y_mix,yh)
            loss = getFussLoss(y_mix,y,yh,self.batch_size)

        # Get average metrics for: sinri, sdri
        sisnri_val, sdri_val = sisnri_sdri(
            y,
            yh,
            y_mix,
            self.batch_size,
            self.num_speakers,
            self.num_speakers,
            pit_axis=1,
        )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        self.sdri_tracker.update_state(sdri_val)
        self.sisnri_tracker.update_state(sisnri_val)
        return {
            "snr_vary_n_source": self.loss_tracker.result()*-1,
            "si-snri": self.sisnri_tracker.result(),
            "sdri": self.sdri_tracker.result(),
        }

    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [
            self.loss_tracker,
            self.sisnri_tracker,
            self.sdri_tracker,
        ]

    def test_step(self, inputs):
        X, y = inputs
        y_mix = tf.math.reduce_sum(y, axis=1, keepdims=True)
        yh = self.model(X)
        yh = enforce_mixture_consistency_time_domain(y_mix,yh)
        loss = getFussLoss(y_mix,y,yh,self.batch_size)

        sisnri_val, sdri_val = sisnri_sdri(
            y,
            yh,
            y_mix,
            self.batch_size,
            self.num_speakers,
            self.num_speakers,
            pit_axis=1,
        )

        self.loss_tracker.update_state(loss)
        self.sdri_tracker.update_state(sdri_val)
        self.sisnri_tracker.update_state(sisnri_val)
        return {
            "snr_vary_n_source": self.loss_tracker.result()*-1,
            "si-snri": self.sisnri_tracker.result(),
            "sdri": self.sdri_tracker.result(),
        }


#########################################
#///////////////////////////////////////#
#########################################


def getRHA(configs,print_model):
    BATCH_SIZE = configs['batch_size']
    MAX_INPUT_LENGTH_IN_SECONDS = configs['max_input_length_in_seconds']
    SAMPLERATE_HZ = configs['samplerate_hz']
    NETWORK_NUM_SOURCES = configs['network_n_sources']
    NETWORK_NUM_FILTERS_IN_ENCODER = configs['network_num_filters_in_encoder']
    NETWORK_ENCODER_FILTER_LENGTH = configs['network_encoder_filter_length']
    NETWORK_NUM_HEAD_PER_ATT = configs['network_num_head_per_att']
    NEWORK_DIM_KEY_ATT = configs['network_dim_key_att']
    NETWORK_NUM_TRAN_BLOCKS = configs['network_num_tran_blocks']
    NETWORK_NUM_CHOP_BLOCKS = configs['network_num_chop_blocks']
    NETWORK_CHUNK_SIZE = configs['network_chunk_size']
    OPTIMIZER_CLIP_L2_NORM_VALUE = 5
    LR = configs['lr']
    NUM_CHUNKS = SAMPLERATE_HZ * MAX_INPUT_LENGTH_IN_SECONDS // NETWORK_CHUNK_SIZE
    num_overlapping_chunks = NUM_CHUNKS * 2 - 1

    model = RHA(
        batch_size=BATCH_SIZE,
        model_weights_file=None,
        num_filters_in_encoder=NETWORK_NUM_FILTERS_IN_ENCODER,
        encoder_filter_length=NETWORK_ENCODER_FILTER_LENGTH,
        chunk_size=NETWORK_CHUNK_SIZE,
        num_full_chunks=NUM_CHUNKS,
        num_chop_blocks=NETWORK_NUM_CHOP_BLOCKS,
        num_tran_blocks=NETWORK_NUM_TRAN_BLOCKS,
        num_head_per_att=NETWORK_NUM_HEAD_PER_ATT,
        dim_key_att=NEWORK_DIM_KEY_ATT,
        max_input_length_in_seconds=MAX_INPUT_LENGTH_IN_SECONDS,
        samplerate_hz=SAMPLERATE_HZ,
        num_speakers=NETWORK_NUM_SOURCES,
    )

    opt = AdamW(LR, clipnorm=OPTIMIZER_CLIP_L2_NORM_VALUE)
    model.compile(optimizer=opt,run_eagerly=True)

    if print_model == True:
        print(model.model.summary())

    return model


#########################################
#///////////////////////////////////////#
#########################################


def getModel(configs,print_model=True):
    if configs['model_type'] == "SepFormer":
        return getSepFormer(configs,print_model)
    elif configs['model_type'] == "RHA":
        return getRHA(configs,print_model)
