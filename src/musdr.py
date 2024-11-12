import numpy as np
import pandas as pd
import scipy.stats


N_PITCH_CLS = 12 # {C, C#, ..., Bb, B}


def compute_histogram_entropy(hist):
    '''
    Computes the entropy (log base 2) of a normalised histogram.

    Parameters:
    hist (ndarray): input pitch (or duration) histogram, should be normalised.

    Returns:
    float: entropy (log base 2) of the histogram.
    '''
    return scipy.stats.entropy(hist) / np.log(2)


def get_pitch_histogram(ev_seq, pitch_evs=range(128), verbose=False):
    '''
    Computes the pitch-class histogram from an event sequence.

    Parameters:
    ev_seq (list): a piece of music in event sequence representation.
    pitch_evs (list): encoding IDs of ``Note-On`` events, should be sorted in increasing order by pitches.
    verbose (bool): whether to print msg. when ev_seq has no notes.

    Returns:
    ndarray: the resulting pitch-class histogram.
    '''
    ev_seq = [x for x in ev_seq if x in pitch_evs]

    if not len(ev_seq):
        if verbose:
            print ('[Info] The sequence contains no notes.')
        return

    # compress sequence to pitch classes & get normalised counts
    ev_seq = pd.Series(ev_seq) % N_PITCH_CLS
    ev_hist = ev_seq.value_counts(normalize=True)

    # make the final histogram
    hist = np.zeros( (N_PITCH_CLS,) )
    for i in range(N_PITCH_CLS):
        if i in ev_hist.index:
            hist[i] = ev_hist.loc[i]
    return hist


def get_onset_xor_distance(seq_a, seq_b, bar_ev_id, pos_evs, pitch_evs=range(128)):
    '''
    Computes the XOR distance of onset positions between a pair of bars.

    Parameters:
    seq_a, seq_b (list): event sequence of a bar of music.
        IMPORTANT: for this implementation, a ``Note-Position`` event must appear before the associated ``Note-On``.
    bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
    pos_evs (list): encoding IDs of ``Note-Position`` events, vocabulary-dependent.
    pitch_evs (list): encoding IDs of ``Note-On`` events.

    Returns:
    float: 0~1, the XOR distance between the 2 bars' (seq_a, seq_b) binary vectors of onsets.
    '''
    # sanity checks
    assert seq_a[0] == bar_ev_id and seq_b[0] == bar_ev_id
    assert seq_a.count(bar_ev_id) == 1 and seq_b.count(bar_ev_id) == 1

    # compute binary onset vectors
    n_pos = len(pos_evs)

    def make_onset_vec(seq):
        cur_pos = -1
        onset_vec = np.zeros((n_pos,))
        for ev in seq:
            if ev in pos_evs:
                cur_pos = ev - pos_evs[0]
            if ev in pitch_evs:
                onset_vec[cur_pos] = 1
        return onset_vec
    a_onsets, b_onsets = make_onset_vec(seq_a), make_onset_vec(seq_b)

    # compute XOR distance
    dist = np.sum( np.abs(a_onsets - b_onsets) ) / n_pos
    return dist


def get_bars_crop(ev_seq, start_bar, end_bar, bar_ev_id, verbose=False):
    '''
    Returns the designated crop (bars) of the input piece.

    Parameter:
    ev_seq (list): a piece of music in event sequence representation.
    start_bar (int): the starting bar of the crop.
    end_bar (int): the ending bar (inclusive) of the crop.
    bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
    verbose (bool): whether to print messages when unexpected operations happen.

    Returns:
    list: a cropped segment of music consisting of (end_bar - start_bar + 1) bars.
    '''
    if start_bar < 0 or end_bar < 0:
        raise ValueError('Invalid start_bar: {}, or end_bar: {}.'.format(start_bar, end_bar))

    # get the indices of ``Bar`` events
    ev_seq = np.array(ev_seq)
    bar_markers = np.where(ev_seq == bar_ev_id)[0]

    if start_bar > len(bar_markers) - 1:
        raise ValueError('start_bar: {} beyond end of piece.'.format(start_bar))

    if end_bar < len(bar_markers) - 1:
        cropped_seq = ev_seq[ bar_markers[start_bar] : bar_markers[end_bar + 1] ]
    else:
        if verbose:
            print (
            '[Info] end_bar: {} beyond or equal the end of the input piece; only the last {} bars are returned.'.format(
                end_bar, len(bar_markers) - start_bar
            ))
        cropped_seq = ev_seq[ bar_markers[start_bar] : ]

    return cropped_seq.tolist()
