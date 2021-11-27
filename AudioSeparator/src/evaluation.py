import tensorflow as tf
import numpy as np

import warnings
import pprint
import functools
import itertools
import typing

from loss import *


coeff = 10.0 / tf.math.log(10.0)
K = 1e-10

def snr(y_true, y_pred):
    noise = y_true - y_pred
    signal_pwr = tf.math.reduce_sum(y_true ** 2, -1)
    noise_pwr = tf.math.reduce_sum(noise ** 2, -1)
    return coeff * (tf.math.log(signal_pwr + K) - tf.math.log(noise_pwr + K))

def sdr(y_true, y_pred):
    s_target = dot(y_true, y_pred) * y_true / (dot(y_true)+K)
    e_all = y_pred - s_target
    s_target_pwr = tf.math.reduce_sum(s_target ** 2, -1)
    e_all_pwr = tf.math.reduce_sum(e_all ** 2, -1)
    return coeff * (tf.math.log(s_target_pwr + K) - tf.math.log(e_all_pwr + K))

    # return coeff * (tf.log(_dot(y_true, y_pred)**2) - \
    #             tf.log(_dot(y_true,y_true) * _dot(y_pred,y_pred) - _dot(y_true, y_pred)**2))

    # return tf.squeeze(coeff * (tf.log(dot(y_true, y_pred)**2) -
    #     tf.log(dot(y_true) * dot(y_pred) - dot(y_true, y_pred)**2)), -1)

def sdr_modify(y_true, y_pred):
    up = tf.math.reduce_sum(y_true * y_pred, -1)
    down = tf.math.sqrt(tf.math.reduce_sum(y_true ** 2, -1) * tf.math.reduce_sum(y_pred ** 2, -1))
    return up / down
    # Ven = tf.reduce_sum(y_true * y_pred, -1) ** 2 / tf.reduce_sum(y_pred ** 2, -1)
    # SDR = tf.sqrt(Ven / tf.reduce_sum(y_true ** 2, -1) )
    # return SDR

def sisnr(y_true, y_pred): # propose by conv-tasnet (which is identity to SDR ??)
    # tasnet : scale invariance is ensured by normalizing s_hat and s to zero-mean prior to the calculation
    y_true = y_true - tf.math.reduce_mean(y_true, -1, keepdims=True)
    y_pred = y_pred - tf.math.reduce_mean(y_pred, -1, keepdims=True)


    s_target = dot(y_true, y_pred) * y_true / (dot(y_true)+K)
    e_noise = y_pred - s_target
    s_target_pwr = tf.math.reduce_sum(s_target ** 2, -1)
    e_noise_pwr = tf.math.reduce_sum(e_noise ** 2, -1)
    return coeff * (tf.math.log(s_target_pwr + K) - tf.math.log(e_noise_pwr + K))

def dot(x,y=None):
    return tf.math.reduce_sum(x * y, -1, keepdims=True) if y != None \
        else tf.math.reduce_sum(x ** 2, -1, keepdims=True)

def _dot(x,y):
    return tf.math.reduce_sum(x*y, -1)

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


def compute_final_metrics(source_waveforms, separated_waveforms, mixture_waveform):
    """Permutation-invariant SI-SNR, powers, and under/equal/over-separation."""
    perm_inv_loss = wrap(lambda tar, est: -signal_to_noise_ratio_gain_invariant(est, tar))
    _, separated_waveforms = perm_inv_loss(source_waveforms,separated_waveforms)

    # Compute separated and source powers.
    power_separated = tf.reduce_mean(separated_waveforms ** 2, axis=-1)
    power_sources = tf.reduce_mean(source_waveforms ** 2, axis=-1)

    # Compute weights for active (separated, source) pairs where source is nonzero
    # and separated power is above threshold of quietest source power - 20 dB.
    weights_active_refs = _weights_for_nonzero_refs(source_waveforms)
    weights_active_seps = _weights_for_active_seps(
        tf.boolean_mask(power_sources, weights_active_refs), power_separated)
    weights_active_pairs = tf.logical_and(weights_active_refs,
                                        weights_active_seps)

    # Compute SI-SNR.
    sisnr_separated = signal_to_noise_ratio_gain_invariant(separated_waveforms, source_waveforms)
    num_active_refs = tf.math.reduce_sum(tf.cast(weights_active_refs, tf.int32))
    num_active_seps = tf.math.reduce_sum(tf.cast(weights_active_seps, tf.int32))
    num_active_pairs = tf.math.reduce_sum(tf.cast(weights_active_pairs, tf.int32))
    sisnr_mixture = signal_to_noise_ratio_gain_invariant(
      tf.tile(mixture_waveform, (1,source_waveforms.shape[1], 1)),source_waveforms)

    # Compute under/equal/over separation.
    under_separation = tf.cast(tf.less(num_active_seps, num_active_refs),
                             tf.float32)
    equal_separation = tf.cast(tf.equal(num_active_seps, num_active_refs),
                             tf.float32)
    over_separation = tf.cast(tf.greater(num_active_seps, num_active_refs),
                            tf.float32)

    return {'sisnr_separated': sisnr_separated,
          'sisnr_mixture': sisnr_mixture,
          'sisnr_improvement': sisnr_separated - sisnr_mixture,
          'power_separated': power_separated,
          'power_sources': power_sources,
          'under_separation': under_separation,
          'equal_separation': equal_separation,
          'over_separation': over_separation,
          'weights_active_refs': weights_active_refs,
          'weights_active_seps': weights_active_seps,
          'weights_active_pairs': weights_active_pairs,
          'num_active_refs': num_active_refs,
          'num_active_seps': num_active_seps,
          'num_active_pairs': num_active_pairs}


def _report_score_stats(metric_per_source_count, label='', counts=None):
    """Report mean and std dev for specified counts."""
    values_all = []
    if counts is None:
        counts = metric_per_source_count.keys()
    for count in counts:
        values = metric_per_source_count[count]
        values_all.extend(list(values))
    return '%s for count(s) %s = %.1f +/- %.1f dB' % (label, counts, np.mean(values_all), np.std(values_all))


def getFinalMetricsFuss(model,test_dataset,EXPERIMENT_NAME,n_samples=1000):
    i=1
    max_count = 4
    dict_per_source_count = lambda: {c: [] for c in range(1, max_count + 1)}
    sisnr_per_source_count = dict_per_source_count()
    sisnri_per_source_count = dict_per_source_count()
    under_seps = []
    equal_seps = []
    over_seps = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        for i in range(n_samples):
            # Get data
            for sample in test_dataset.take(1):
                mix_wave, source_wave = sample

            # Pass through model
            sep_wave = model(mix_wave)

            # Reshape mix wave
            mix_wave = tf.expand_dims(mix_wave,axis=0)

            # Get metrics
            metrics_dict = metrics = compute_final_metrics(source_wave,sep_wave,mix_wave)

            metrics_dict = {k: v.numpy() for k, v in metrics_dict.items()}
            sisnr_sep = metrics_dict['sisnr_separated']
            sisnr_mix = metrics_dict['sisnr_mixture']
            sisnr_imp = metrics_dict['sisnr_improvement']
            weights_active_pairs = metrics_dict['weights_active_pairs']

            # Store metrics per source count and report results so far.
            under_seps.append(metrics_dict['under_separation'])
            equal_seps.append(metrics_dict['equal_separation'])
            over_seps.append(metrics_dict['over_separation'])
            sisnr_per_source_count[metrics_dict['num_active_refs']].extend(
            sisnr_sep[weights_active_pairs].tolist())
            sisnri_per_source_count[metrics_dict['num_active_refs']].extend(
                    sisnr_imp[weights_active_pairs].tolist())

            # Report mean statistics and save csv every so often.
            lines = [
                      'Metrics after %d examples:' % (i+1),
                      _report_score_stats(sisnr_per_source_count, 'SI-SNR',
                                          counts=[1]),
                      _report_score_stats(sisnri_per_source_count, 'SI-SNRi',
                                          counts=[2]),
                      _report_score_stats(sisnri_per_source_count, 'SI-SNRi',
                                          counts=[3]),
                      _report_score_stats(sisnri_per_source_count, 'SI-SNRi',
                                          counts=[4]),
                      _report_score_stats(sisnri_per_source_count, 'SI-SNRi',
                                          counts=[2, 3, 4]),
                      'Under separation: %.2f' % np.mean(under_seps),
                      'Equal separation: %.2f' % np.mean(equal_seps),
                      'Over separation: %.2f' % np.mean(over_seps),
            ]

        print('')
        for line in lines:
            print(line)

        with open('log/'+EXPERIMENT_NAME+'/'+EXPERIMENT_NAME+'_summary.txt', 'w') as f:
            f.writelines([line + '\n' for line in lines])
