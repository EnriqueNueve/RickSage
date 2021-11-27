import numpy as np
import tensorflow as tf

import warnings
import pprint
import functools
import itertools
import typing


import math
from itertools import permutations
import evaluation

#########################################
#///////////////////////////////////////#
#########################################


def _resolve_permutation(loss_matrix):
    """Resolves permutation from an all-pairs loss_matrix input.

    Args:
        loss_matrix: tensor of shape [batch, source, source]
            axis 1 refers to the estimate.
            axis 2 refers to the reference.
    Returns:
    permutation: tensor of shape [batch, source, 2] such that
        tf.gather_nd(estimates, permutation, 1) returns the permuted estimates
        that achieves the lowest loss.
    """
    batch = loss_matrix.shape[0]
    source = loss_matrix.shape[1]

    # Compute permutations as vectors of indices into flattened loss matrix.
    # permutations will have shape [batch, source!, source, 1].
    permutations = tf.constant(list(itertools.permutations(range(source))))
    permutations = tf.expand_dims(tf.expand_dims(permutations, 0), 3)
    permutations = tf.tile(permutations, [batch, 1, 1, 1])

    # Expand loss dimensions for gather.
    # loss_matrix.shape will be (batch, source!, source, source)
    loss_matrix = tf.expand_dims(loss_matrix, 1)
    loss_matrix = tf.tile(loss_matrix, [1, permutations.shape[1], 1, 1])

    # Compute the total loss for each permutation.
    # permuted_loss.shape will be (batch, source!)
    permuted_loss = tf.gather_nd(loss_matrix, permutations, batch_dims=3)
    permuted_loss = tf.math.reduce_sum(permuted_loss, axis=2)

    # Get and return the permutation with the lowest total loss.
    # loss_argmin.shape will be (batch, 1)
    loss_argmin = tf.math.argmin(permuted_loss, axis=1)
    loss_argmin = tf.expand_dims(loss_argmin, 1)

    # permutation.shape will be (batch, source, 1)
    permutation = tf.gather_nd(permutations, loss_argmin, batch_dims=1)

    return permutation


def _apply(loss_fn: typing.Callable[..., tf.Tensor],
           reference: tf.Tensor,
           estimate: tf.Tensor,
           allow_repeated: bool,
           enable: bool) -> typing.Any:
    """Return permutation invariant loss.

    Note that loss_fn must in general handle an arbitrary number of sources, since
    this function may expand in that dimention to get losses on all
    reference-estimate pairs.

    Args:
        loss_fn: function with the following signature:
        Args
            reference [batch, source', ...] tensor
            estimate [batch, source', ...] tensor
    Returns
        A [batch, source'] tensor of dtype=tf.float32
        reference: [batch, source, ...] tensor.
        estimate: [batch, source, ...] tensor.
        allow_repeated: If true, allow the same estimate to be used to match
          multiple references.
        enable: If False, apply the loss function in fixed order and return its
          value and the unpermuted estimates.

    Returns:
        loss, A [batch, source] tensor of dtype=tf.float32
        permuted_estimate, A tensor like estimate.
    """
    reference = tf.convert_to_tensor(reference)
    estimate = tf.convert_to_tensor(estimate)

    if not enable:
        return loss_fn(reference, estimate), estimate

    assert reference.shape[:2] == estimate.shape[:2]
    batch = reference.shape[0]
    source = reference.shape[1]

    # Replicate estimate on axis 1
    # estimate.shape will be (batch, source * source, ...)
    multiples = np.ones_like(estimate.shape)
    multiples[1] = source
    estimate_tiled = tf.tile(estimate, multiples)

    # Replicate reference on new axis 2, then combine axes [1, 2].
    # reference.shape will be (batch, source * source, ...)
    reference_tiled = tf.expand_dims(reference, 2)
    multiples = np.ones_like(reference_tiled.shape)
    multiples[2] = source
    reference_tiled = tf.tile(reference_tiled, multiples)
    reference_tiled = tf.reshape(reference_tiled, estimate_tiled.shape)

    # Compute the loss matrix.
    # loss_matrix.shape will be (batch, source, source).
    # Axis 1 is the estimate.  Axis 2 is the reference.
    loss_matrix = tf.reshape(loss_fn(reference_tiled, estimate_tiled),
                           [batch, source, source])

    # Get the best permutation.
    # permutation.shape will be (batch, source, 1)
    if allow_repeated:
        permutation = tf.math.argmin(loss_matrix, axis=2, output_type=tf.int32)
        permutation = tf.expand_dims(permutation, 2)
    else:
        permutation = _resolve_permutation(loss_matrix)
    assert permutation.shape == (batch, source, 1), permutation.shape

    # Permute the estimates according to the best permutation.
    estimate_permuted = tf.gather_nd(estimate, permutation, batch_dims=1)
    loss_permuted = tf.gather_nd(loss_matrix, permutation, batch_dims=2)

    return loss_permuted, estimate_permuted


def wrap(loss_fn: typing.Callable[..., tf.Tensor],
         allow_repeated: bool = False,
         enable: bool = True) -> typing.Callable[..., typing.Any]:
    """Returns a permutation invariant version of loss_fn.

    Args:
        loss_fn: function with the following signature:
        Args
            reference [batch, source', ...] tensor
            estimate [batch, source', ...] tensor
            **args Any remaining arguments to loss_fn
    Returns
        A [batch, source'] tensor of dtype=tf.float32
    allow_repeated: If true, allow the same estimate to be used to match
      multiple references.
    enable: If False, return a fuction that applies the loss function in fixed
      order, returning its value and the (unpermuted) estimate.

    Returns:
        A function with same arguments as loss_fn returning loss, permuted_estimate
    """
    def wrapped_loss_fn(reference, estimate, **args):
        return _apply(functools.partial(loss_fn, **args),
                  reference,
                  estimate,
                  allow_repeated,
                  enable)
    return wrapped_loss_fn


def calculate_signal_to_noise_ratio_from_power(signal_power, noise_power, epsilon):
    """Computes the signal to noise ratio given signal_power and noise_power.

    Args:
    signal_power: A tensor of unknown shape and arbitrary rank.
    noise_power: A tensor matching the signal tensor.
    epsilon: An optional float for numerical stability, since silences
        can lead to divide-by-zero.

    Returns:
        A tensor of size [...] with SNR computed between matching slices of the
        input signal and noise tensors.
    """
    # Pre-multiplication and change of logarithm base.
    constant = tf.cast(10.0 / tf.math.log(10.0), signal_power.dtype)

    return constant * tf.math.log(tf.math.truediv(signal_power + epsilon, noise_power + epsilon))


def calculate_signal_to_noise_ratio(signal, noise, epsilon=1e-8):
    """Computes the signal to noise ratio given signal and noise.

    Args:
        signal: A [..., samples] tensor of unknown shape and arbitrary rank.
        noise: A tensor matching the signal tensor.
        epsilon: An optional float for numerical stability, since silences
          can lead to divide-by-zero.

    Returns:
          A tensor of size [...] with SNR computed between matching slices of the
        input signal and noise tensors.
    """
    def power(x):
        return tf.math.reduce_mean(tf.square(source_wave), axis=-1)


    return calculate_signal_to_noise_ratio_from_power(power(signal), power(noise), epsilon)


def signal_to_noise_ratio_gain_invariant(estimate, target, epsilon=1e-8):
    """Computes the signal to noise ratio in a gain invariant manner.

      This computes SNR assuming that the signal equals the target multiplied by an
      unknown gain, and that the noise is orthogonal to the target.

      This quantity is also known as SI-SDR [1, equation 5].

      This function estimates SNR using a formula given e.g. in equation 4.38 from
      [2], which gives accurate results on a wide range of inputs, and yields a
      monotonically decreasing value when target or estimate scales toward zero.

      [1] Jonathan Le Roux, Scott Wisdom, Hakan Erdogan, John R. Hershey,
      "SDR--half-baked or well done?",ICASSP 2019,
      https://arxiv.org/abs/1811.02508.
      [2] Magnus Borga, "Learning Multidimensional Signal Processing"
      https://www.diva-portal.org/smash/get/diva2:302872/FULLTEXT01.pdf

      Args:
        estimate: An estimate of the target of size [..., samples].
        target: A ground truth tensor, matching estimate above.
        epsilon: An optional float introduced for numerical stability in the
          projections only.

      Returns:
        A tensor of size [...] with SNR computed between matching slices of the
        input signal and noise tensors.
    """
    def normalize(x):
        power = tf.math.reduce_sum(tf.square(x), keepdims=True, axis=[-1])
        return tf.math.multiply(x, tf.math.rsqrt(tf.math.maximum(power, 1e-16)))

    normalized_estimate = normalize(estimate)
    normalized_target = normalize(target)

    cosine_similarity = tf.math.reduce_sum(tf.math.multiply(normalized_estimate, normalized_target),axis=[-1])
    squared_cosine_similarity = tf.math.square(cosine_similarity)
    normalized_signal_power = squared_cosine_similarity
    normalized_noise_power = 1. - squared_cosine_similarity

    # Computing normalized_noise_power as the difference between very close
    # floating-point numbers is not accurate enough for this case, so when
    # normalized_signal power is close to 0., we use an alternate formula.
    # Both formulas are accurate enough at the 'seam' in float32.
    normalized_noise_power_direct = tf.math.reduce_sum(
              tf.math.square(normalized_estimate -
                normalized_target * tf.expand_dims(cosine_similarity, -1)),axis=[-1])

    normalized_noise_power = tf.where(
        tf.greater_equal(normalized_noise_power, 0.01),
        normalized_noise_power,
        normalized_noise_power_direct)

    return calculate_signal_to_noise_ratio_from_power(
        normalized_signal_power, normalized_noise_power, epsilon)


def signal_to_noise_ratio_residual(estimate, target, epsilon=1e-8):
    """Computes the signal to noise ratio using residuals.

    This computes the SNR in a "statistical fashion" as the logarithm of the
    relative residuals. The signal is defined as the original target, and the
    noise is the residual between the estimate and the target. This is
    proportional to log(1 - 1/R^2).

    Args:
        estimate: An estimate of the target of size [..., samples].
        target: A ground truth tensor, matching estimate above.
        epsilon: An optional float for numerical stability, since silences
        can lead to divide-by-zero.

    Returns:
        A tensor of size [...] with SNR computed between matching slices of the
        input signal and noise tensors.
    """
    return calculate_signal_to_noise_ratio(target, target - estimate, epsilon=epsilon)



def _weights_for_nonzero_refs(source_waveforms):
    """Return shape (source,) weights for signals that are nonzero."""
    source_norms = tf.sqrt(tf.reduce_mean(tf.square(source_waveforms), axis=-1))
    return tf.greater(source_norms, 1e-8)


def _weights_for_active_seps(power_sources, power_separated):
    """Return (source,) weights for active separated signals."""
    min_power = tf.reduce_min(power_sources, axis=-1, keepdims=True)
    return tf.greater(power_separated, 0.01 * min_power)


def _stabilized_log_base(x, base=10., stabilizer=1e-8):
    """Stabilized log with specified base."""
    logx = tf.math.log(x + stabilizer)
    logb = tf.math.log(tf.constant(base, dtype=logx.dtype))
    return logx / logb

def log_mse_loss(source, separated, max_snr=1e6, bias_ref_signal=None):
    """Negative log MSE loss, the negated log of SNR denominator."""
    err_pow = tf.math.reduce_sum(tf.math.square(source - separated), axis=-1)
    snrfactor = 10.**(-max_snr / 10.)
    if bias_ref_signal is None:
        ref_pow = tf.math.reduce_sum(tf.square(source), axis=-1)
    else:
        ref_pow = tf.math.reduce_sum(tf.math.square(bias_ref_signal), axis=-1)
    bias = snrfactor * ref_pow
    return 10. * _stabilized_log_base(bias + err_pow)


def groupwise_apply(loss_fns: typing.Dict[str, typing.Callable[..., typing.Any]],
          signal_names: typing.List[str],
          reference: tf.Tensor,
          estimate: tf.Tensor,
          permutation_invariant_losses: typing.List[str]):
    """Apply loss functions to the corresponding references and estimates.

    For each kind of signal, gather corresponding references and estimates, and
    apply the loss function.  Scatter-add the results into the loss.

    For elements of signals_names not in loss_fns, no loss will be applied.

    Args:
        loss_fns: dictionary of string -> loss_fn.
            Each string is a name to match elements of signal_names.
            Each loss_fn has the following signature:
        Args
            reference [batch, grouped_source, ...] tensor
            estimate [batch, grouped_source, ...] tensor
    Returns
        A [batch, grouped_source] tensor of dtype=tf.float32
        signal_names: list of names of each signal.
        reference: [batch, source, ...] tensor.
        estimate: [batch, source, ...] tensor.
        permutation_invariant_losses: List of losses to be permutation invariant.

    Returns:
        loss, A [batch, source] tensor of dtype=tf.float32
    """
    if reference.shape[:2] != estimate.shape[:2]:
        raise ValueError('First two axes (batch, source) of reference and estimate'
                     'must be equal, got {}, {}'.format(
                         reference.shape[:2], estimate.shape[:2]))

    batch = reference.shape[0]
    loss = tf.zeros(shape=reference.shape[:2], dtype=tf.float32)
    permuted_estimates = tf.zeros_like(reference)

    # For each kind of signal, e.g. 'speech', 'noise', gather subsets of reference
    # and estimate, apply loss function and scatter-add into the loss tensor.
    for name, loss_fn in loss_fns.items():

        idxs = [idx for idx, value in enumerate(signal_names) if value == name]
        idxs_0 = tf.tile(
            tf.expand_dims(tf.range(batch), 1),
            [1, len(idxs)])
        idxs_1 = tf.tile(
            tf.expand_dims(tf.constant(idxs, dtype=tf.int32), 0),
            [batch, 1])

        idxs_nd = tf.stack([idxs_0, idxs_1], axis=2)
        reference_key = tf.gather_nd(reference, idxs_nd)
        estimate_key = tf.gather_nd(estimate, idxs_nd)

        loss_fn = wrap(
            loss_fn,
            enable=name in permutation_invariant_losses)
        loss_key, permuted_estimates_key = loss_fn(reference_key, estimate_key)

        loss = tf.tensor_scatter_nd_add(loss, idxs_nd, loss_key)
        permuted_estimates = tf.tensor_scatter_nd_add(
            permuted_estimates, idxs_nd, permuted_estimates_key)

    return loss, permuted_estimates


def getFussLoss(mixture_waveforms,source_waveforms,separated_waveforms,batch_size):
    hparams_signal_types = ['source'] * 4
    unique_signal_types = list(set(hparams_signal_types))
    loss_fns = {signal_type: log_mse_loss for signal_type in unique_signal_types}

    _, separated_waveforms = groupwise_apply(loss_fns,
                                            hparams_signal_types,
                                            source_waveforms,
                                            separated_waveforms,
                                            unique_signal_types)

    # Build loss split between all-zero and nonzero reference signals.
    source_is_nonzero = _weights_for_nonzero_refs(source_waveforms)
    source_is_zero = tf.math.logical_not(source_is_nonzero)

    # Get batch size and (max) number of sources.
    num_sources = 4

    # Waveforms with nonzero references.
    source_waveforms_nonzero = tf.boolean_mask(
          source_waveforms, source_is_nonzero)[:, tf.newaxis]
    separated_waveforms_nonzero = tf.boolean_mask(
          separated_waveforms, source_is_nonzero)[:, tf.newaxis]

    # Waveforms with all-zero references.
    source_waveforms_zero = tf.boolean_mask(
          source_waveforms, source_is_zero)[:, tf.newaxis]
    separated_waveforms_zero = tf.boolean_mask(
          separated_waveforms, source_is_zero)[:, tf.newaxis]

    weight = 1. / tf.cast(batch_size * num_sources, tf.float32)

    mixture_waveforms_zero = tf.boolean_mask(
            tf.tile(mixture_waveforms[:, 0:1], (1, num_sources, 1)),
            source_is_zero)[:, tf.newaxis]
    loss = tf.math.reduce_sum(log_mse_loss(source_waveforms_zero,
                                          separated_waveforms_zero,
                                          max_snr=20,
                                          bias_ref_signal=mixture_waveforms_zero))
    loss_zero = tf.identity(1 * weight * loss, name='loss_ref_zero')

    # Loss for nonzero references.
    loss = tf.math.reduce_sum(log_mse_loss(source_waveforms_nonzero,
                                        separated_waveforms_nonzero,
                                        max_snr=30))
    loss_nonzero = tf.identity(weight * loss, name='loss_ref_nonzero')

    return loss_zero+loss_nonzero


#########################################
#///////////////////////////////////////#
#########################################




def pit_loss(y_true, y_pred, loss_type, batch_size, n_speaker, n_output, pit_axis=1):
    # [batch, spk #, length]
    real_spk_num = n_speaker

    # TODO 1: # output channel != # speaker
    v_perms = tf.constant(list(permutations(range(n_output), n_speaker)))
    v_perms_onehot = tf.one_hot(v_perms, n_output)

    y_true_exp = tf.expand_dims(
        y_true, pit_axis + 1
    )  # [batch, n_speaker, 1,        len]
    y_pred_exp = tf.expand_dims(y_pred, pit_axis)  # [batch, 1,         n_output, len]

    cross_total_loss = get_loss(loss_type, y_true_exp, y_pred_exp)

    loss_sets = tf.einsum("bij,pij->bp", cross_total_loss, v_perms_onehot)
    loss = tf.reduce_min(loss_sets, axis=1)
    loss = tf.reduce_mean(loss)

    # find permutation sets for y pred
    s_perm_sets = tf.argmin(loss_sets, 1)
    s_perm_choose = tf.gather(v_perms, s_perm_sets)
    s_perm_idxs = tf.stack(
        [
            tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, n_speaker]),
            s_perm_choose,
        ],
        axis=2,
    )

    s_perm_idxs = tf.reshape(s_perm_idxs, [batch_size * n_speaker, 2])
    y_pred = tf.gather_nd(y_pred, s_perm_idxs)
    y_pred = tf.reshape(y_pred, [batch_size, n_speaker, -1])

    if loss_type != "sdr":
        sdr = evaluation.sdr(y_true[:, :real_spk_num, :], y_pred[:, :real_spk_num, :])
        sdr = tf.reduce_mean(sdr)
    else:
        sdr = -loss / n_speaker

    return loss, y_pred, sdr, s_perm_choose


def get_loss(loss_type, t_true_exp, t_pred_exp, axis=-1):
    if loss_type == "l1":
        y_cross_loss = t_true_exp - t_pred_exp
        cross_total_loss = tf.reduce_sum(tf.abs(y_cross_loss), axis=axis)

    elif loss_type == "l2":
        y_cross_loss = t_true_exp - t_pred_exp
        cross_total_loss = tf.reduce_sum(tf.square(y_cross_loss), axis=axis)

    elif loss_type == "snr":
        cross_total_loss = -evaluation.snr(t_true_exp, t_pred_exp)

    elif loss_type == "sdr":
        cross_total_loss = -evaluation.sdr(t_true_exp, t_pred_exp)

    elif loss_type == "sisnr":
        cross_total_loss = -evaluation.sisnr(t_true_exp, t_pred_exp)

    elif loss_type == "sdr_modify":
        cross_total_loss = -evaluation.sdr_modify(t_true_exp, t_pred_exp)

    elif loss_type == "sisdr":
        cross_total_loss = -evaluation.sisdr(t_true_exp, t_pred_exp)

    elif loss_type == "sym_sisdr":
        cross_total_loss = -evaluation.sym_sisdr(t_true_exp, t_pred_exp)

    return cross_total_loss


#########################################
# ///////////////////////////////////////#
#########################################


@tf.function
def sisnri_sdri(s, s_est, mix_s, batch_size, n_speaker, n_output, pit_axis=1):
    mix_s = tf.repeat(mix_s, n_speaker, axis=1)
    mix_s = tf.reshape(mix_s, (batch_size, n_speaker, -1))

    loss, _, sdr, _ = pit_loss(
        s, s_est, "sisnr", batch_size, n_speaker, n_output, pit_axis=1
    )
    loss_b, _, sdr_b, _ = pit_loss(
        s, mix_s, "sisnr", batch_size, n_speaker, n_output, pit_axis=1
    )

    loss *= -1
    loss_b *= -1

    '''
    if tf.math.is_nan(loss_b) and tf.math.is_nan(sdr_b) == False:
        print('a')
        return loss * -1, sdr - sdr_b
    elif tf.math.is_nan(loss_b) == False and tf.math.is_nan(sdr_b):
        print('b')
        return loss - loss_b, sdr_b
    elif tf.math.is_nan(loss_b) and tf.math.is_nan(sdr_b):
        print('c')
        return loss * -1, sdr
    else:
        return loss - loss_b, sdr - sdr_b
    '''
    return loss - loss_b, sdr - sdr_b



#########################################
#///////////////////////////////////////#
#########################################


"""
@tf.function
def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

@tf.function
def sisnr(s, s_hat, do_expand=False, eps=1e-8):
    if do_expand:
        s = np.expand_dims(s, axis=0)
        s_hat = np.expand_dims(s_hat, axis=0)
    dot_product = tf.math.reduce_sum(s*s_hat, axis=1, keepdims=True)
    squares = tf.math.reduce_sum(s*s, axis=1, keepdims=True)
    s_target = s * dot_product / squares
    e_noise = s_hat - s_target
    s_target_squared = tf.math.reduce_sum(s_target*s_target, axis=1)
    e_noise_squared = tf.math.reduce_sum(e_noise*e_noise, axis=1)
    return 10*log10(s_target_squared / (e_noise_squared + eps))

@tf.function
def permutation_invariant_loss(y_true, y_pred,n_sources=2):
    sn, sn_hat = [], []
    for i in range(n_sources):
        sn.append(y_true[:,i,:])
        sn_hat.append(y_pred[:,i,:])

    if n_sources == 2:
        sisnr_perm0_spk0 = sisnr(sn[0], sn_hat[0])
        sisnr_perm0_spk1 = sisnr(sn[1], sn_hat[1])
        sisnr_perm0 = (sisnr_perm0_spk0 + sisnr_perm0_spk1) / 2

        sisnr_perm1_spk0 = sisnr(sn[0], sn_hat[1])
        sisnr_perm1_spk1 = sisnr(sn[1], sn_hat[0])
        sisnr_perm1 = (sisnr_perm1_spk0 + sisnr_perm1_spk1) / 2

        sisnr_perm_invariant = tf.math.maximum(sisnr_perm0, sisnr_perm1)
        return -sisnr_perm_invariant

    elif n_sources == 3:
        sisnr_perm0_spk0 = sisnr(sn[0], sn_hat[0])
        sisnr_perm0_spk1 = sisnr(sn[1], sn_hat[1])
        sisnr_perm0_spk2 = sisnr(sn[2], sn_hat[2])
        sisnr_perm0 = (sisnr_perm0_spk0 + sisnr_perm0_spk1 + sisnr_perm0_spk2) / 3

        sisnr_perm1_spk0 = sisnr(sn[0], sn_hat[0])
        sisnr_perm1_spk1 = sisnr(sn[1], sn_hat[2])
        sisnr_perm1_spk2 = sisnr(sn[2], sn_hat[1])
        sisnr_perm1 = (sisnr_perm1_spk0 + sisnr_perm1_spk1 + sisnr_perm1_spk2) / 3

        sisnr_perm2_spk0 = sisnr(sn[0], sn_hat[1])
        sisnr_perm2_spk1 = sisnr(sn[1], sn_hat[0])
        sisnr_perm2_spk2 = sisnr(sn[2], sn_hat[2])
        sisnr_perm2 = (sisnr_perm2_spk0 + sisnr_perm2_spk1 + sisnr_perm2_spk2) / 3

        sisnr_perm3_spk0 = sisnr(sn[0], sn_hat[1])
        sisnr_perm3_spk1 = sisnr(sn[1], sn_hat[2])
        sisnr_perm3_spk2 = sisnr(sn[2], sn_hat[0])
        sisnr_perm3 = (sisnr_perm3_spk0 + sisnr_perm3_spk1 + sisnr_perm3_spk2) / 3

        sisnr_perm4_spk0 = sisnr(sn[0], sn_hat[2])
        sisnr_perm4_spk1 = sisnr(sn[1], sn_hat[0])
        sisnr_perm4_spk2 = sisnr(sn[2], sn_hat[1])
        sisnr_perm4 = (sisnr_perm4_spk0 + sisnr_perm4_spk1 + sisnr_perm4_spk2) / 3

        sisnr_perm5_spk0 = sisnr(sn[0], sn_hat[2])
        sisnr_perm5_spk1 = sisnr(sn[1], sn_hat[1])
        sisnr_perm5_spk2 = sisnr(sn[2], sn_hat[0])
        sisnr_perm5 = (sisnr_perm5_spk0 + sisnr_perm5_spk1 + sisnr_perm5_spk2) / 3

        sisnr_perm_invariant = tf.stack([sisnr_perm1,sisnr_perm2,sisnr_perm3,sisnr_perm4,sisnr_perm5])
        sisnr_perm_invariant = tf.math.reduce_max(sisnr_perm_invariant)
        return -sisnr_perm_invariant
"""

"""
@tf.function
def get_permutation_invariant_sisnr(spk0_estimate, spk1_estimate, spk0_groundtruth, spk1_groundtruth):
    perm0_spk0 = sisnr(spk0_groundtruth, spk0_estimate, do_expand=True)
    perm0_spk1 = sisnr(spk1_groundtruth, spk1_estimate, do_expand=True)
    perm1_spk0 = sisnr(spk0_groundtruth, spk1_estimate, do_expand=True)
    perm1_spk1 = sisnr(spk1_groundtruth, spk0_estimate, do_expand=True)

    # Get best permutation
    if perm0_spk0 + perm0_spk1 > perm1_spk0 + perm1_spk1:
        return perm0_spk0, perm0_spk1

    return perm1_spk0, perm1_spk1


@tf.function
def permutation_invariant_loss(y_true, y_pred,n_sources=2):
    # PIT for n-sources, work but very slow, needs to use tf.while_loop
    # yet, implementing tf.while_loop is a nightmare
    sn, sn_hat = [], []
    for i in range(n_sources):
        sn.append(y_true[:,i,:])
        sn_hat.append(y_pred[:,i,:])

    perm = list(permutations(range(n_sources)))
    sisnr_perm = []

    for i in range(len(perm)):
        sisnr_perm_spk = [sisnr(sn[p], sn_hat[j]) for p,j in enumerate(perm[i])]
        sisnr_perm.append(tf.math.reduce_sum(sisnr_perm_spk)/n_sources)

    sisnr_perm = tf.stack(sisnr_perm)
    return -tf.math.reduce_max(sisnr_perm)
"""
