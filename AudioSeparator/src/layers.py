import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math


#########################################
#///////////////////////////////////////#
#########################################


def enforce_mixture_consistency_time_domain(mixture_waveforms,
                                            separated_waveforms):
    """Projection implementing mixture consistency in time domain.
        This projection makes the sum across sources of separated_waveforms equal
        mixture_waveforms and minimizes the unweighted mean-squared error between the
        sum across sources of separated_waveforms and mixture_waveforms. See
        https://arxiv.org/abs/1811.08521 for the derivation.
        Args:
            mixture_waveforms: Tensor of mixture waveforms in waveform format.
            separated_waveforms: Tensor of separated waveforms in source image format.
        Returns:
            Projected separated_waveforms as a Tensor in source image format.
    """
    # Modify the source estimates such that they sum up to the mixture, where
    # the mixture is defined as the sum across sources of the true source
    # targets. Uses the least-squares solution under the constraint that the
    # resulting source estimates add up to the mixture.

    num_sources = 4.0
    mix_estimate = tf.reduce_sum(separated_waveforms, axis=1, keepdims=True)
    #mix_weights = tf.reduce_mean(tf.square(separated_waveforms), axis=[1, 2],keepdims=True)
    #mix_weights /= tf.reduce_sum(mix_weights, axis=1, keepdims=True)
    mix_weights = (1.0 / num_sources)
    mix_weights = tf.cast(mix_weights, tf.float32)
    correction = mix_weights * (mixture_waveforms - mix_estimate)
    separated_waveforms = separated_waveforms + correction

    return separated_waveforms


#########################################
#///////////////////////////////////////#
#########################################


class ChunkOperator(layers.Layer):
    """Performs a chunk operation on a 2D (feature,time) tensor
    and outputs a 3D (feature,short_time,long_time) tensor.

    Parameters
    ----------
    samplerate_hz : int
        The samlerate of the audio clip
    max_input_length_in_seconds : int
        The max length of input audio in seconds, if clip is shorter than
        max length then zero padding is applied
    chunk_length : int
        States length of sliding window in chunk operation over time axis
    num_filters_in_encoder: int
        Number of filters in Conv1d operation used before chunk operation, this
        data is needed for dim/padding problems
    batch_size : int
        The batch size is needed for reshaping during the chunk operation

    Returns
    -------
    tf.constant
        3D (feature,short_time,long_time) tensor
    """

    def __init__(
        self,
        samplerate_hz,
        max_input_length_in_seconds,
        chunk_length,
        num_filters_in_encoder,
        batch_size,
    ):
        super(ChunkOperator, self).__init__()

        # Constants
        self.num_filters_in_encoder = num_filters_in_encoder
        self.batch_size = batch_size
        self.samplerate_hz = samplerate_hz
        self.max_input_length_in_seconds = max_input_length_in_seconds
        self.chunk_length = chunk_length

        # Dependants
        self.num_full_chunks = (
            self.samplerate_hz * self.max_input_length_in_seconds // self.chunk_length
        )
        self.signal_length_samples = self.chunk_length * self.num_full_chunks
        self.chunk_advance = self.chunk_length // 2
        self.num_overlapping_chunks = self.num_full_chunks * 2 - 1

        # The layer itself
        self.chunk_operator = tf.keras.layers.Lambda(self.segment_encoded_signal)

    @tf.function
    def segment_encoded_signal(self, x):
        x1 = tf.reshape(
            x,
            (
                self.batch_size,
                self.signal_length_samples // self.chunk_length,
                self.chunk_length,
                self.num_filters_in_encoder,
            ),
        )
        x2 = tf.roll(x, shift=-self.chunk_advance, axis=1)
        x2 = tf.reshape(
            x2,
            (
                self.batch_size,
                self.signal_length_samples // self.chunk_length,
                self.chunk_length,
                self.num_filters_in_encoder,
            ),
        )
        x2 = x2[:, :-1, :, :]  # Discard last segment with invalid data

        x_concat = tf.concat([x1, x2], axis=1)
        x = x_concat[:, :: self.num_full_chunks, :, :]
        for i in range(1, self.num_full_chunks):
            x = tf.concat([x, x_concat[:, i :: self.num_full_chunks, :, :]], axis=1)
        return x

    @tf.function
    def call(self, x):
        x = self.chunk_operator(x)
        return x


#########################################
#///////////////////////////////////////#
#########################################


class Encoder(layers.Layer):
    """Takes original audio input x -> ReLU(Conv1D(x)) -> h0
         -> LayerNorm(Dense(h0)) -> h1 -> Chunk(h1) -> hc

    Parameters
    ----------
    num_filters_in_encoder: int
        Number of filters in Conv1d operation used before chunk operation, this
        data is needed for dim/padding problems
    encoder_filter_length: int
        Length of filter in Conv1D
    samplerate_hz : int
        The samlerate of the audio clip
    max_input_length_in_seconds : int
        The max length of input audio in seconds, if clip is shorter than
        max length then zero padding is applied
    chunk_length : int
        States length of sliding window in chunk operation over time axis
    batch_size : int
        The batch size is needed for reshaping during the chunk operation

    Returns
    -------
    tf.constant
        3D (feature,short_time,long_time) tensor
    """

    def __init__(
        self,
        num_filters_in_encoder,
        encoder_filter_length,
        samplerate_hz,
        max_input_length_in_seconds,
        chunk_length,
        batch_size,
    ):
        super(Encoder, self).__init__()

        # Constants
        self.batch_size = batch_size
        self.num_filters_in_encoder = num_filters_in_encoder
        self.encoder_filter_length = encoder_filter_length
        self.batch_size = batch_size
        self.samplerate_hz = samplerate_hz
        self.max_input_length_in_seconds = max_input_length_in_seconds
        self.chunk_length = chunk_length

        # Dependants
        self.encoder_hop_size = self.encoder_filter_length // 2

    def build(self, input_shape):
        self.conv1d = tf.keras.layers.Conv1D(
            filters=self.num_filters_in_encoder,
            kernel_size=self.encoder_filter_length,
            strides=self.encoder_hop_size,
            use_bias=False,
            padding="same",
        )
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.chunk_operator = ChunkOperator(
            samplerate_hz=self.samplerate_hz,
            max_input_length_in_seconds=self.max_input_length_in_seconds,
            chunk_length=self.chunk_length,
            num_filters_in_encoder=self.num_filters_in_encoder,
            batch_size=self.batch_size,
        )

    @tf.function
    def call(self, inputs):
        encoder_out = self.conv1d(tf.expand_dims(inputs, axis=2))
        x = self.layer_norm(encoder_out)
        x = self.chunk_operator(x)
        return x, encoder_out


#########################################
#///////////////////////////////////////#
#########################################


class SineSPE(layers.Layer):
    def __init__(
        self,
        num_heads: int = 8,
        in_features: int = 64,
        num_realizations: int = 256,
        num_sines: int = 1,
    ):
        super(SineSPE, self).__init__()

        self.num_heads = num_heads
        self.in_features = in_features
        self.num_sines = num_sines
        self.num_realizations = num_realizations

        freqs_init = tf.random_normal_initializer()
        self.freqs = tf.Variable(
            initial_value=freqs_init(
                shape=(num_heads, in_features, num_sines), dtype="float32"
            ),
            trainable=True,
        )

        offsets_init = tf.random_normal_initializer()
        self.offsets = tf.Variable(
            initial_value=offsets_init(
                shape=(num_heads, in_features, num_sines), dtype="float32"
            ),
            trainable=True,
        )

        gains_init = tf.random_normal_initializer()
        self.gains = tf.Variable(
            initial_value=gains_init(
                shape=(num_heads, in_features, num_sines), dtype="float32"
            ),
            trainable=True,
        )

        # Normalize gains
        self.gains = self.gains / (
            tf.math.sqrt(tf.norm(self.gains, axis=-1, keepdims=True)) / 2
        )

        # Bias intial freqs
        self.freqs = self.freqs - 4

        self.code_shape = (num_heads, in_features)

    def call(self, shape):
        """
        Generate the code, composed of a random QBar and Kbar,
        depending on the parameters, and return them for use with a
        SPE module to actually encode queries and keys.
        Args:
            shape: The outer shape of the inputs: (batchsize, *size)
            num_realizations: if provided, overrides self.num_realizations
        """

        if len(shape) != 2:
            raise ValueError("Only 1D inputs are supported by SineSPE")

        max_len = shape[1]

        # build omega_q and omega_k
        # with shape (num_heads,keys_dim,length,2*num_sines)
        indices = tf.linspace(0, max_len - 1, max_len)
        indices = tf.cast(indices, dtype=tf.float32)

        # make sure freqs are in [0,.5]
        freqs = tf.nn.sigmoid(self.freqs[:, :, None, :]) / 2

        phases_q = (
            2
            * math.pi
            * freqs
            * indices[None, None, :, None]
            * self.offsets[:, :, None, :]
        )
        omega_q = tf.stack([tf.math.cos(phases_q), tf.math.sin(phases_q)], axis=-1)
        omega_q = tf.reshape(
            omega_q, [1, self.num_heads, self.in_features, max_len, 2 * self.num_sines]
        )

        phases_k = 2 * math.pi * freqs * indices[None, None, :, None]
        omega_k = tf.stack([tf.math.cos(phases_k), tf.math.sin(phases_k)], axis=-1)
        omega_k = tf.reshape(
            omega_k, [1, self.num_heads, self.in_features, max_len, 2 * self.num_sines]
        )

        # Gains is (num_heads,keys_dim,num_sines), make nonnegative with softplut
        gains = tf.math.softplus(self.gains)

        # Upsample
        gains = tf.stack([gains, gains], axis=-1)
        gains = tf.reshape(
            gains, [self.num_heads, self.in_features, 2 * self.num_sines]
        )

        # Draw noise
        z = tf.random.normal(
            (
                1,
                self.num_heads,
                self.in_features,
                2 * self.num_sines,
                self.num_realizations,
            )
        )
        z = z / tf.math.sqrt(tf.cast(self.num_sines * 2, dtype=tf.float32))

        # Scale each of the 2*num_sines by the appropriate gain
        z = z * gains[None, ..., None]

        # Compute sums over sines
        qbar = tf.linalg.matmul(omega_q, z)
        kbar = tf.linalg.matmul(omega_k, z)

        # Pemute to (1,length,num_heads,key_dim,num_realization)
        qbar = tf.transpose(qbar, perm=[0, 3, 1, 2, 4])
        kbar = tf.transpose(kbar, perm=[0, 3, 1, 2, 4])

        # scale
        scale = (self.num_realizations * self.in_features) ** 0.25
        return (qbar / scale, kbar / scale)


#########################################
#///////////////////////////////////////#
#########################################


class SPEFilter(layers.Layer):
    """Stochastic positional encoding filter
    Applies a positional code provided by a SPE module on actual queries and keys.
    Implements gating, i.e. some "dry" parameter, that lets original queries and keys through if activated.
    Args:
    gated: whether to use the gated version, which learns to balance
        positional and positionless features.
    code_shape: the inner shape of the codes, i.e. (num_heads, key_dim),
        as given by `spe.code_shape`
    """

    def __init__(self, gated, code_shape):
        super(SPEFilter, self).__init__()

        self.gated = gated
        self.code_shape = code_shape

        # create the gating parameters if required
        if gated:
            if code_shape is None:
                raise RuntimeError("code_shape has to be provided if gated is True.")

            gate_init = tf.random_normal_initializer()
            self.gate = tf.Variable(
                initial_value=gate_init(shape=(code_shape), dtype="float32"),
                trainable=True,
            )

    def call(self, queries, keys, code):
        """
        Apply SPE on keys with a given code.
        Expects keys and queries of shape `(batch_size, ..., num_heads,
        key_dim)` and outputs keys and queries of shape `(batch_size,
        ..., num_heads, num_realizations)`. code is the tuple
        of the 2 tensors provided by the code instance, each one of
        shape (1, ..., num_heads, key_dim, num_realizations)
        """
        assert queries.shape == keys.shape, (
            "As of current implementation, queries and keys must have the same shape. "
            "got queries: {} and keys: {}".format(queries.shape, keys.shape)
        )

        # qbar and kbar are (1, *shape, num_heads, keys_dim, num_realizations)
        (qbar, kbar) = code

        # check that codes have the shape we are expecting
        if self.code_shape is not None and qbar.shape[-3:-1] != self.code_shape:
            raise ValueError(
                f"The inner shape of codes is {qbar.shape[-3:-1]}, "
                f"but expected {self.code_shape}"
            )

        # check shapes: size of codes should be bigger than queries, keys
        code_size = qbar.shape[1:-3]
        query_size = queries.shape[1:-2]

        # if (len(code_size) != len(query_size)
        #    or tf.reduce_any(
        #        tf.Variable(code_size) < tf.Variable(query_size)
        #    )):
        #        raise ValueError(f'Keys/queries have length {query_size}, '
        #                         f'but expected at most {code_size}')

        if len(code_size) != len(query_size):
            raise ValueError(
                f"Keys/queries have length {query_size}, "
                f"but expected at most {code_size}"
            )

        if qbar.shape[-3:-1] != queries.shape[-2:]:
            raise ValueError(
                f"shape mismatch. codes have shape {qbar.shape}, "
                f"but queries are {queries.shape}"
            )

        # truncate qbar and kbar for matching current queries and keys,
        # but only if we need to
        for dim in range(len(query_size)):
            if code_size[dim] > query_size[dim]:
                indices = [
                    slice(1),
                    *[slice(qbar.shape[1 + k]) for k in range(dim)],
                    slice(query_size[dim]),
                ]
                qbar = qbar[indices]
                kbar = kbar[indices]

        # apply gate if required
        if self.gated:
            # incorporate the constant bias for Pd if required. First draw noise
            # such that noise noise^T = 1, for each head, feature, realization.
            # qbar is : (1, *shape, num_heads, keys_dim, num_realizations)
            in_features = qbar.shape[-2]
            num_realizations = qbar.shape[-1]
            gating_noise = (
                tf.random.normal(self.code_shape + (num_realizations,))
                / (in_features * num_realizations) ** 0.25
            )

            # normalize it so that it's an additive 1 to Pd
            # gating_noise = gating_noise / gating_noise.norm(dim=2, keepdim=True)

            # constrain the gate parameter to be in [0 1]
            gate = tf.math.sigmoid(self.gate[..., None])

            # qbar is (1, *shape, num_heads, keys_dim, num_realizations)
            # gating noise is (num_heads, keys_dim, num_realizations)
            # gate is (num_heads, keys_dim, 1)
            # import ipdb; ipdb.set_trace()
            qbar = tf.math.sqrt(1.0 - gate) * qbar + tf.math.sqrt(gate) * gating_noise
            kbar = tf.math.sqrt(1.0 - gate) * kbar + tf.math.sqrt(gate) * gating_noise

        # sum over d after multiplying by queries and keys
        # qbar/kbar are (1, *shape, num_heads, keys_dim, num_realizations)
        # queries/keys  (batchsize, *shape, num_heads, keys_dim)
        qhat = tf.math.reduce_sum(qbar * queries[..., None], axis=-2)
        khat = tf.math.reduce_sum(kbar * keys[..., None], axis=-2)

        # result is (batchsize, ..., num_heads, num_realizations)
        return (qhat, khat)


#########################################
#///////////////////////////////////////#
#########################################


@tf.function
def compute_linear_mhsa(q, k, v):
    q = tf.nn.gelu(q) + 1  # Needed for kernel assumption
    k = tf.nn.gelu(k) + 1
    kv = tf.einsum("... h s d, ...  h s m  -> ... h m d", k, v)
    k_sum = tf.math.reduce_sum(k, axis=2)
    z = 1 / (tf.einsum("... h l d, ... h d -> ... h l", q, k_sum) + 1e-4)
    Vhat = tf.einsum("... h l d, ... h m d, ... h l -> ... h l m", q, kv, z)
    return Vhat


class LinearAttentionSineSPE(tf.keras.layers.Layer):
    def __init__(self, d_model, heads=8, num_sines=5):
        super(LinearAttentionSineSPE, self).__init__()
        self.num_heads = heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

        self.spe_encoder = SineSPE(
            num_heads=heads,  # Number of attention heads
            in_features=self.depth,  # Dimension of keys and queries
            num_realizations=self.depth,  # New dimension of keys and queries
            num_sines=num_sines,
        )  # Number of sinusoidal components
        self.spe_filter = SPEFilter(gated=True, code_shape=self.spe_encoder.code_shape)

    @tf.function
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])

        pos_codes = self.spe_encoder(q.shape[:2])  # pos_codes is a tuple (qbar, kbar)
        q, k = self.spe_filter(q, k, pos_codes)
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention = compute_linear_mhsa(q, k, v)

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output


#########################################
#///////////////////////////////////////#
#########################################


class LinearSineSPETransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):

        super(LinearSineSPETransformerBlock, self).__init__()

        self.lha = LinearAttentionSineSPE(embed_dim, num_heads)

        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="gelu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.lha(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


#########################################
#///////////////////////////////////////#
#########################################


class Decoder(layers.Layer):
    """ Takes Chop Shop output and combines with decoder output
        to form sepeated sources.

    Parameters
    ----------
    n_sources: int
        Number of sources to sepeate audio into
    num_filters_in_encoder: int
        Number of filters in Conv1d operation used before chunk operation, this
        data is needed for dim/padding problems
    encoder_filter_length: int
        Length of filter in Conv1D
    samplerate_hz : int
        The samlerate of the audio clip
    max_input_length_in_seconds : int
        The max length of input audio in seconds, if clip is shorter than
        max length then zero padding is applied
    chunk_length : int
        States length of sliding window in chunk operation over time axis
    batch_size : int
        The batch size is needed for reshaping during the chunk operation

    Returns
    -------
    tf.constant
        List of seperated signals
    """

    def __init__(self,signal_length_samples,
                 n_sources,
                 num_filters_in_encoder,
                 encoder_filter_length,
                 samplerate_hz,
                 max_input_length_in_seconds,
                 chunk_length,
                 batch_size):
        super(Decoder,self).__init__()

        # Constants
        self.n_sources = n_sources
        self.batch_size = batch_size
        self.num_filters_in_encoder = num_filters_in_encoder
        self.encoder_filter_length = encoder_filter_length
        self.batch_size = batch_size
        self.samplerate_hz = samplerate_hz
        self.signal_length_samples = signal_length_samples
        self.max_input_length_in_seconds = max_input_length_in_seconds
        self.chunk_length = chunk_length
        self.chunk_advance =  chunk_length  // 2

        # Dependants
        self.encoder_hop_size = self.encoder_filter_length // 2

        self.cut_len = 0
        if self.num_filters_in_encoder%self.n_sources != 0:
            self.cut_len = self.num_filters_in_encoder%self.n_sources


    def build(self,input_shape):
        self.overlap_and_add_mask_segments_layer = keras.layers.Lambda(self.overlap_and_add_mask_segments)

        self.DenseLayers = []
        for i in range(self.n_sources):
            dense_layer = keras.layers.Dense(units=self.encoder_filter_length, use_bias=False,name='OverLapAddDense_'+str(i))
            self.DenseLayers.append(dense_layer)

        self.OverLapAndAddDecoderLayers = []
        for i in range(self.n_sources):
            overlap_and_add_in_decoder_layer = keras.layers.Lambda(self.overlap_and_add_in_decoder,name='OverLapAdd_'+str(i))
            self.OverLapAndAddDecoderLayers.append(overlap_and_add_in_decoder_layer)

    @tf.function
    def overlap_and_add_mask_segments(self, x):
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.signal.overlap_and_add(x, self.chunk_advance)
        return tf.transpose(x, [0, 2, 1])

    @tf.function
    def overlap_and_add_in_decoder(self, x):
        return tf.signal.overlap_and_add(x, self.encoder_hop_size)

    @tf.function
    def call(self, x, encoder_out):
        masks = self.overlap_and_add_mask_segments_layer(x)
        x = masks*encoder_out

        if self.cut_len != 0:
            x = tf.split(x,num_or_size_splits=[self.num_filters_in_encoder-self.cut_len,self.cut_len], axis=-1, num=None, name='split_cut')[0]
        d_sources = tf.split(x, num_or_size_splits=self.n_sources, axis=-1, num=None, name='split')

        decoded_sources = []
        for i in range(self.n_sources):
            decoded_spk = self.DenseLayers[i](d_sources[i])
            decoded_spk = self.OverLapAndAddDecoderLayers[i](decoded_spk)[:, :self.signal_length_samples]
            decoded_sources.append(decoded_spk)

        decoded = tf.stack(decoded_sources, axis=1)

        return decoded
