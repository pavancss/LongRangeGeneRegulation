"""
Single-nucleus transcription analysis for MS2/PP7 live-imaging time courses.

Conventions used throughout this module
---------------------------------------
* A *trace* is a 1D sequence of per-frame focus intensities for one nucleus.
  Within one embryo all traces have the same length (one value per frame).
* Embryos are usually imaged for different numbers of frames, so traces and
  per-embryo summaries from different embryos have different lengths. Before
  they can be stacked into one table they must be *aligned*:

    - ``align='right'`` (default, original behaviour): series share their LAST
      frame. Shorter series are padded with NaN at the front. Use this when the
      end of the movie is the common reference point (e.g. gastrulation /
      end of NC14).
    - ``align='left'``: series share their FIRST frame. Shorter series are
      padded with NaN at the end. Use this when the start of the movie is the
      common reference point (e.g. the NC13/NC14 mitosis).

  Within-embryo functions (``singlenuc_ints``, ``nucCount``, ``normNucCount``,
  ``burst_metrics`` ...) do not need an alignment choice, because every trace in
  an embryo already spans the same frames.
"""

import os

import numpy as np
import pandas as pd
from skimage.measure import label


_VALID_ALIGN = ('left', 'right')
_VALID_CENSOR = ('end', 'start', 'both', 'none')


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

def _check_align(align):
    if align not in _VALID_ALIGN:
        raise ValueError(f"align must be one of {_VALID_ALIGN}, got {align!r}")


def pad_series(series, length, align='right', fill_value=np.nan):
    """
    Pad one series to ``length`` so it lines up with others on one side.

    Parameters
    ----------
    series : sequence of float
        The series to pad (list, tuple or 1D array).
    length : int
        Target length. Must be >= ``len(series)``.
    align : {'right', 'left'}, default 'right'
        'right' pads at the front so the last values line up;
        'left' pads at the end so the first values line up.
    fill_value : float, default np.nan
        Value used for padding.

    Returns
    -------
    list
        Padded copy of ``series`` (the input is not modified).
    """
    _check_align(align)
    series = list(series)
    n_pad = length - len(series)
    if n_pad < 0:
        raise ValueError(f"series of length {len(series)} is longer than target length {length}")
    pad = [fill_value] * n_pad
    return pad + series if align == 'right' else series + pad


def align_series(list_of_series, align='right', length=None, fill_value=np.nan):
    """
    Stack series of different lengths into a 2D array, aligned on one side.

    Parameters
    ----------
    list_of_series : list of sequences
        One series per row (e.g. one per embryo or one per nucleus).
    align : {'right', 'left'}, default 'right'
        Which end the series share. See module docstring.
    length : int, optional
        Number of columns. Defaults to the longest series.
    fill_value : float, default np.nan
        Value used for padding.

    Returns
    -------
    np.ndarray, shape (len(list_of_series), length)
    """
    if len(list_of_series) == 0:
        return np.empty((0, length or 0))
    if length is None:
        length = max(len(s) for s in list_of_series)
    return np.array([pad_series(s, length, align, fill_value) for s in list_of_series],
                    dtype=float)


def aligned_time_axis(n_frames, time_per_image, align='right', total_time=60, start_time=0):
    """
    Build the time axis (one value per frame) for a set of aligned series.

    Parameters
    ----------
    n_frames : int
        Number of frames (columns) in the aligned table.
    time_per_image : float
        Frame interval (e.g. minutes).
    align : {'right', 'left'}, default 'right'
        'right': the last frame is assigned ``total_time`` and earlier frames
        count back from it. 'left': the first frame is assigned ``start_time``
        and later frames count forward.
    total_time : float, default 60
        Time of the last frame when ``align='right'``.
    start_time : float, default 0
        Time of the first frame when ``align='left'``.

    Returns
    -------
    list of float, length n_frames
    """
    _check_align(align)
    if time_per_image is None:
        raise ValueError("time_per_image must be provided")
    if align == 'right':
        return [total_time - time_per_image * (n_frames - 1 - i) for i in range(n_frames)]
    return [start_time + time_per_image * i for i in range(n_frames)]


def _n_frames(burstInt_singlenuc):
    """Length of the traces in a single-nucleus dict (all traces share it)."""
    return len(next(iter(burstInt_singlenuc.values())))


# ---------------------------------------------------------------------------
# Per-embryo extraction and summaries
# ---------------------------------------------------------------------------

def singlenuc_ints(tracked_nucs_filter, G_foci_int=np.zeros((1, 1)), R_foci_int=np.zeros((1, 1))):
    """
    Sum focus intensity inside each tracked nucleus, frame by frame.

    Parameters
    ----------
    tracked_nucs_filter : np.ndarray, shape (T, Y, X), int
        Nuclear label stack where a nucleus keeps the same label over time
        (output of ``nucs_border_time_filter``).
    G_foci_int : np.ndarray, shape (T, Y, X), optional
        Background-subtracted MS2 focus intensity image (``ms2_segmentation``).
        Leave as the default to skip the MS2 channel.
    R_foci_int : np.ndarray, shape (T, Y, X), optional
        Background-subtracted PP7 focus intensity image (``pp7_segmentation``).
        Leave as the default to skip the PP7 channel.

    Returns
    -------
    If both channels are given: ``(gburstInt_singlenuc, rburstInt_singlenuc)``.
    If only one channel is given: that channel's dict.
    Each dict maps nuclear label -> list of length T with the summed focus
    intensity of that nucleus in each frame.

    Notes
    -----
    - Frames where the nucleus has no focus AND frames where the nucleus is not
      present at all (before it appears / after it is lost) are both 0; no NaN
      is written by this function.
    - Labels are iterated as ``range(1, max_label)``, so the highest label is
      not included. Kept as is so results match previous analyses.
    """
    gburstInt_singlenuc = {}
    rburstInt_singlenuc = {}
    n_frames = tracked_nucs_filter.shape[0]

    if not np.array_equal(G_foci_int, np.zeros((1, 1))):
        for nuclabel in range(1, np.max(tracked_nucs_filter)):
            nuc_mask = tracked_nucs_filter == nuclabel
            foci_g = G_foci_int * nuc_mask
            gburstInt_singlenuc[nuclabel] = [np.sum(foci_g[t, :, :]) for t in range(n_frames)]

    if not np.array_equal(R_foci_int, np.zeros((1, 1))):
        for nuclabel in range(1, np.max(tracked_nucs_filter)):
            nuc_mask = tracked_nucs_filter == nuclabel
            foci_r = R_foci_int * nuc_mask
            rburstInt_singlenuc[nuclabel] = [np.sum(foci_r[t, :, :]) for t in range(n_frames)]

    if len(gburstInt_singlenuc) > 0 and len(rburstInt_singlenuc) > 0:
        return gburstInt_singlenuc, rburstInt_singlenuc
    elif len(gburstInt_singlenuc) > 0:
        return gburstInt_singlenuc
    elif len(rburstInt_singlenuc) > 0:
        return rburstInt_singlenuc
    else:
        print('Provide non-empty label masks for foci')
        return np.array([]), np.array([])


def nucCount(gburstInt_singlenuc, tracked_nucs_filter):
    """
    Count transcriptionally active nuclei over time and total signal per frame.

    Parameters
    ----------
    gburstInt_singlenuc : dict
        Nuclear label -> trace of summed focus intensity (``singlenuc_ints``).
        All traces span the same frames; NaN entries are treated as 0.
    tracked_nucs_filter : np.ndarray, shape (T, Y, X)
        Labeled nuclear masks; used only to count nuclei present per frame.

    Returns
    -------
    tcourse : np.ndarray, shape (T,)
        Number of nuclei with intensity > 0 at each frame.
    normed_tcourse : np.ndarray, shape (T,)
        ``tcourse`` as a percentage of the nuclei present in that frame.
    G_instexp : list of float, length T
        Summed intensity across all active nuclei at each frame.

    Notes
    -----
    - A nucleus counts as positive if its trace sums to > 0 over the movie.
    - The per-frame denominator is ``len(np.unique(mask_t))``, which includes
      the background label 0, so percentages are slightly deflated.
    - The denominator changes over time as nuclei are gained or lost.
    - All outputs are per-frame within one embryo; align across embryos later
      with ``create_timeseries_dataframe(..., align=...)``.
    """
    n_frames = _n_frames(gburstInt_singlenuc)

    G_pos_nucs = {}
    for nuc, trace in gburstInt_singlenuc.items():
        g_ints = np.nan_to_num(np.array(trace, dtype=float))
        if np.sum(g_ints) > 0:
            G_pos_nucs[nuc] = g_ints

    g_ints_mat = np.zeros((len(G_pos_nucs), n_frames))
    for ctr, nuc in enumerate(G_pos_nucs):
        g_ints_mat[ctr, :] = G_pos_nucs[nuc]

    tot_nucs = [len(np.unique(tracked_nucs_filter[t, :, :])) for t in range(tracked_nucs_filter.shape[0])]
    tcourse = np.sum(g_ints_mat > 0, axis=0)
    normed_tcourse = 100 * (tcourse / tot_nucs)
    G_instexp = [np.sum(g_ints_mat[:, t]) for t in range(n_frames)]

    return tcourse, normed_tcourse, G_instexp


def normNucCount(gburstInt_singlenuc, rburstInt_singlenuc):
    """
    PP7 (chrb) activity among MS2 (scyl)-positive nuclei, frame by frame.

    Only nuclei whose MS2 trace is > 0 at some point in the movie are used, so
    PP7 measures are normalized to cells that saw the enhancer.

    Parameters
    ----------
    gburstInt_singlenuc, rburstInt_singlenuc : dict
        Nuclear label -> trace of summed MS2 / PP7 focus intensity
        (``singlenuc_ints``). Must share keys and trace length. NaN -> 0.

    Returns
    -------
    normed_tcourse : np.ndarray, shape (T,)
        % of MS2-positive nuclei with a PP7 focus at each frame.
    cum_tcourse : list of float, length T
        % of MS2-positive nuclei that had a PP7 focus in any EARLIER frame
        (frames ``0 .. t-1``; the current frame is excluded, so entry 0 is 0).
    scyl_instexp : list of float
        Summed MS2 intensity across MS2-positive nuclei at each frame.
    chrb_instexp : list of float
        Summed PP7 intensity across MS2-positive nuclei at each frame.
    scyl_fociInt : list of float
        Median MS2 intensity of active foci at each frame (NaN if none).
    chrb_fociInt : list of float
        Median PP7 intensity of active foci at each frame (NaN if none).

    Notes
    -----
    The cumulative curve is counted from the first frame of THIS embryo's
    movie. If embryos are aligned on the right and started at different times,
    their cumulative curves have different starting points; aligning on the
    left (``align='left'``) is usually more natural for ``cum_tcourse``.
    """
    n_frames = _n_frames(gburstInt_singlenuc)

    scyl_pos_nucs = {}
    chrb_pos_nucs = {}
    for nuc in gburstInt_singlenuc:
        g_ints = np.nan_to_num(np.array(gburstInt_singlenuc[nuc], dtype=float))
        r_ints = np.nan_to_num(np.array(rburstInt_singlenuc[nuc], dtype=float))
        if np.sum(g_ints) > 0:
            scyl_pos_nucs[nuc] = g_ints
            chrb_pos_nucs[nuc] = r_ints

    tot_scyl_posnucs = len(scyl_pos_nucs)
    r_ints_mat = np.zeros((tot_scyl_posnucs, n_frames))
    g_ints_mat = np.zeros((tot_scyl_posnucs, n_frames))
    for ctr, nuc in enumerate(scyl_pos_nucs):
        r_ints_mat[ctr, :] = chrb_pos_nucs[nuc]
        g_ints_mat[ctr, :] = scyl_pos_nucs[nuc]

    pos_nucs = r_ints_mat > 0
    tcourse = np.sum(pos_nucs, axis=0)
    normed_tcourse = 100 * (tcourse / tot_scyl_posnucs)

    cum_tcourse, scyl_instexp, chrb_instexp, scyl_fociInt, chrb_fociInt = [], [], [], [], []
    for t in range(n_frames):
        col_sum = np.sum(pos_nucs[:, 0:t], axis=1)
        scyl_t = g_ints_mat[:, t]
        chrb_t = r_ints_mat[:, t]
        scyl_instexp.append(np.sum(scyl_t))
        scyl_fociInt.append(np.median(scyl_t[np.nonzero(scyl_t)]))
        chrb_instexp.append(np.sum(chrb_t))
        chrb_fociInt.append(np.median(chrb_t[np.nonzero(chrb_t)]))
        cum_tcourse.append(100 * (np.sum(col_sum > 0) / tot_scyl_posnucs))

    return normed_tcourse, cum_tcourse, scyl_instexp, chrb_instexp, scyl_fociInt, chrb_fociInt


def positive_nucs(tracked_nucs_filter, foci_mask):
    """
    List which nuclei contain a focus in each frame.

    Parameters
    ----------
    tracked_nucs_filter : np.ndarray, shape (T, Y, X)
        Labeled nuclear masks.
    foci_mask : np.ndarray, shape (T, Y, X)
        Focus mask (binary or labeled); any value > 0 counts as a focus.

    Returns
    -------
    pos_nucs : dict
        Frame index -> array of nuclear labels overlapping a focus.
    num_pos_nucs : list of int
        Length of each ``pos_nucs`` entry.

    Notes
    -----
    ``np.unique`` also returns the background label 0 whenever a frame has any
    pixel outside a focus, so ``num_pos_nucs`` is usually one higher than the
    number of positive nuclei.
    """
    pos_nucs = {}
    for t in range(tracked_nucs_filter.shape[0]):
        nucs = tracked_nucs_filter[t, :, :]
        pos_nucs[t] = np.unique(nucs * (foci_mask[t] > 0))
    num_pos_nucs = [len(pos_nucs[t]) for t in range(len(pos_nucs))]
    return pos_nucs, num_pos_nucs


# ---------------------------------------------------------------------------
# Burst (ON) and pause (OFF) statistics
# ---------------------------------------------------------------------------

def _is_censored(run, censor):
    """True if a run touches a movie edge that ``censor`` says to drop."""
    touches_start = bool(run[0])
    touches_end = bool(run[-1])
    if censor == 'end':
        return touches_end
    if censor == 'start':
        return touches_start
    if censor == 'both':
        return touches_start or touches_end
    return False


def _trace_bursts(trace, censor='end'):
    """
    ON lengths, OFF lengths and burst sizes for one trace.

    ON runs are dropped according to ``censor``; OFF runs touching either movie
    edge are always dropped (their true length is unknown).
    """
    intensities = np.array(trace, dtype=float)
    bursts = intensities > 0                      # NaN compares False -> OFF
    burst_lens, burst_sizes, pause_lens = [], [], []

    burst_label = label(bursts)
    for lab in range(1, np.max(burst_label) + 1):
        burst = burst_label == lab
        if not _is_censored(burst, censor):
            burst_lens.append(np.sum(burst))
            burst_sizes.append(np.sum(intensities[burst]))

    pause_label = label(~bursts)
    for lab in range(1, np.max(pause_label) + 1):
        pause = pause_label == lab
        if pause[0] or pause[-1]:
            continue
        pause_lens.append(np.sum(pause))

    return burst_lens, pause_lens, burst_sizes


def burst_metrics(burstInt_singlenuc, censor='end'):
    """
    Burst ON durations, OFF durations and burst sizes for all nuclei in an embryo.

    Parameters
    ----------
    burstInt_singlenuc : dict
        Nuclear label -> trace of focus intensity (``singlenuc_ints``).
    censor : {'end', 'start', 'both', 'none'}, default 'end'
        Which incompletely observed bursts to drop:
        'end'   - drop bursts still ON in the last frame (original behaviour;
                  matches right-aligned movies whose start is the reference),
        'start' - drop bursts already ON in the first frame,
        'both'  - drop bursts touching either edge,
        'none'  - keep every burst.

    Returns
    -------
    burst_lens : list of int
        ON durations in frames, in the order encountered (unsorted).
    pause_lens : list of int
        OFF durations in frames. OFF periods touching the first or last frame
        are always excluded.
    burst_sizes : list of float
        Summed intensity of each burst; matches ``burst_lens`` element-wise.

    Notes
    -----
    Nuclei are iterated as ``range(1, max_label)`` (highest label skipped),
    kept as is so results match previous analyses.
    """
    if censor not in _VALID_CENSOR:
        raise ValueError(f"censor must be one of {_VALID_CENSOR}, got {censor!r}")

    burst_lens, pause_lens, burst_sizes = [], [], []
    for nucctr in range(1, max(burstInt_singlenuc.keys())):
        b, p, s = _trace_bursts(burstInt_singlenuc[nucctr], censor)
        burst_lens += b
        pause_lens += p
        burst_sizes += s
    return burst_lens, pause_lens, burst_sizes


def rburst_metrics(burstInt_singlenuc, gburstInt_singlenuc, censor='end'):
    """
    Same as ``burst_metrics`` but only for nuclei that are MS2-positive.

    Use for PP7 (chrb) bursts restricted to nuclei where MS2 (scyl) was active
    at some point in the movie.

    Parameters
    ----------
    burstInt_singlenuc : dict
        Nuclear label -> PP7 trace.
    gburstInt_singlenuc : dict
        Nuclear label -> MS2 trace (same keys). Nuclei whose MS2 trace sums to
        0 (after NaN -> 0) are skipped.
    censor : {'end', 'start', 'both', 'none'}, default 'end'
        See ``burst_metrics``.

    Returns
    -------
    burst_lens, pause_lens, burst_sizes : lists
        See ``burst_metrics``.
    """
    if censor not in _VALID_CENSOR:
        raise ValueError(f"censor must be one of {_VALID_CENSOR}, got {censor!r}")

    burst_lens, pause_lens, burst_sizes = [], [], []
    for nucctr in range(1, max(burstInt_singlenuc.keys())):
        g_ints = np.nan_to_num(np.array(gburstInt_singlenuc[nucctr], dtype=float))
        if np.sum(g_ints) > 0:
            b, p, s = _trace_bursts(burstInt_singlenuc[nucctr], censor)
            burst_lens += b
            pause_lens += p
            burst_sizes += s
    return burst_lens, pause_lens, burst_sizes


# ---------------------------------------------------------------------------
# Filtering single-nucleus traces
# ---------------------------------------------------------------------------

def ms2_pos_lists(gburstInt_singlenuc):
    """
    Keep only MS2-positive nuclei.

    Parameters
    ----------
    gburstInt_singlenuc : dict
        Nuclear label -> MS2 trace.

    Returns
    -------
    dict
        Original nuclear label -> MS2 trace rounded to integers (``np.rint``),
        for nuclei with at least one frame > 0 (NaN treated as 0).
    """
    gburstIntFilt = {}
    for nuc, trace in gburstInt_singlenuc.items():
        g_ints = np.nan_to_num(np.array(trace, dtype=float))
        if np.sum(g_ints > 0):
            gburstIntFilt[nuc] = np.rint(trace)
    return gburstIntFilt


def scyl_pos_lists(gburstInt_singlenuc, rburstInt_singlenuc):
    """
    PP7 traces of nuclei that are MS2 (scyl)-positive.

    Parameters
    ----------
    gburstInt_singlenuc : dict
        Nuclear label -> MS2 trace; decides which nuclei are kept.
    rburstInt_singlenuc : dict
        Nuclear label -> PP7 trace (same keys).

    Returns
    -------
    dict
        New consecutive index (0, 1, 2, ...) -> PP7 trace rounded to integers,
        for nuclei whose MS2 trace has at least one frame > 0.
        Note the keys are re-indexed, unlike ``ms2_pos_lists``.
    """
    newkey = 0
    rburstIntFilt = {}
    for nuc, trace in gburstInt_singlenuc.items():
        g_ints = np.nan_to_num(np.array(trace, dtype=float))
        if np.sum(g_ints > 0):
            rburstIntFilt[newkey] = np.rint(rburstInt_singlenuc[nuc])
            newkey += 1
    return rburstIntFilt


# ---------------------------------------------------------------------------
# Cross-embryo tables
# ---------------------------------------------------------------------------

def _mean_sem_with_min_count(df_sub, cols, min_n):
    """Row-wise mean and SEM over ``cols``; NaN where fewer than ``min_n`` values."""
    count = df_sub[cols].count(axis=1)
    mean = df_sub[cols].mean(axis=1, skipna=True)
    sem = df_sub[cols].std(axis=1, ddof=1, skipna=True) / count ** 0.5
    mean[count < min_n] = np.nan
    sem[count < min_n] = np.nan
    return mean, sem


def _save_tsv(df, output_path, index):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_csv(output_path, sep='\t', index=index)


def create_timeseries_dataframe(
    ctrl_lists,
    ctrl_identifiers,
    aux_lists=None,
    aux_identifiers=None,
    time_per_image=None,
    total_time=60,
    output_path=None,
    column_prefix='normed',
    min_count=3,
    align='right',
    start_time=0,
):
    """
    Combine per-embryo time series (ctrl and optional aux) into one aligned table.

    Parameters
    ----------
    ctrl_lists : list of sequences
        One time series per control embryo (e.g. ``normed_tcourse``).
    ctrl_identifiers : list of str
        Names for the control embryos, same order as ``ctrl_lists``.
    aux_lists : list of sequences, optional
        One time series per auxin (or other condition) embryo.
    aux_identifiers : list of str, optional
        Names for the aux embryos.
    time_per_image : float
        Frame interval (minutes).
    total_time : float, default 60
        Time assigned to the last row when ``align='right'``.
    output_path : str, optional
        If given, the table is written there as tab-separated text.
    column_prefix : str, default 'normed'
        Per-embryo columns are named ``f'{column_prefix}_{identifier}'``.
    min_count : int, default 3
        Minimum number of embryos with data in a row to report mean/SEM.
    align : {'right', 'left'}, default 'right'
        'right': embryos share their last frame (shorter series NaN-padded at
        the front; original behaviour). 'left': embryos share their first
        frame (shorter series NaN-padded at the end).
    start_time : float, default 0
        Time assigned to the first row when ``align='left'``.

    Returns
    -------
    pd.DataFrame
        Columns: Time | <prefix>_<ctrl ids>... | ctrl_mean | ctrl_sem
                 [| <prefix>_<aux ids>... | aux_mean | aux_sem]  (if aux given)
        Rows where every group mean is NaN (fewer than ``min_count`` embryos in
        all groups) are dropped.

    Notes
    -----
    Ctrl and aux series are padded to a shared length so the groups stay on the
    same time axis.
    """
    _check_align(align)
    aux_lists = aux_lists or []
    aux_identifiers = aux_identifiers or []
    has_aux = len(aux_lists) > 0

    max_len = max(len(s) for s in list(ctrl_lists) + list(aux_lists))

    ctrl_cols = [f'{column_prefix}_{k}' for k in ctrl_identifiers]
    aux_cols = [f'{column_prefix}_{k}' for k in aux_identifiers] if has_aux else []

    data = {c: pad_series(s, max_len, align) for c, s in zip(ctrl_cols, ctrl_lists)}
    if has_aux:
        data.update({c: pad_series(s, max_len, align) for c, s in zip(aux_cols, aux_lists)})
    df = pd.DataFrame(data)

    df['Time'] = aligned_time_axis(max_len, time_per_image, align, total_time, start_time)

    df['ctrl_mean'], df['ctrl_sem'] = _mean_sem_with_min_count(df, ctrl_cols, min_count)
    drop_subset = ['ctrl_mean']
    final_cols = ['Time'] + ctrl_cols + ['ctrl_mean', 'ctrl_sem']

    if has_aux:
        df['aux_mean'], df['aux_sem'] = _mean_sem_with_min_count(df, aux_cols, min_count)
        drop_subset.append('aux_mean')
        final_cols += aux_cols + ['aux_mean', 'aux_sem']

    df = df.dropna(subset=drop_subset, how='all').reset_index(drop=True)
    df = df[final_cols]

    if output_path is not None:
        _save_tsv(df, output_path, index=True)

    return df


def traces_to_dataframe(trace_dicts, identifiers, align='right', id_col='emb',
                        time_prefix='T', output_path=None):
    """
    Put single-nucleus traces from several embryos into one aligned table.

    Replaces the pad-and-stack code in the "Burst Data" notebook cells.

    Parameters
    ----------
    trace_dicts : list of dict
        One dict per embryo, nucleus key -> trace (e.g. output of
        ``scyl_pos_lists`` or ``ms2_pos_lists``).
    identifiers : list of str
        Embryo names, same order as ``trace_dicts``.
    align : {'right', 'left'}, default 'right'
        Which end the traces share across embryos. 'right' reproduces the
        previous NaN front-padding.
    id_col : str, default 'emb'
        Name of the embryo column.
    time_prefix : str, default 'T'
        Frame columns are named ``T1, T2, ...``.
    output_path : str, optional
        If given, written as tab-separated text (no index).

    Returns
    -------
    pd.DataFrame
        One row per nucleus: ``id_col`` followed by one column per frame.
    """
    _check_align(align)
    ids, traces = [], []
    for emb, d in zip(identifiers, trace_dicts):
        for trace in d.values():
            ids.append(emb)
            traces.append(trace)

    if not traces:
        df = pd.DataFrame(columns=[id_col])
    else:
        mat = align_series(traces, align)
        values_df = pd.DataFrame(mat, columns=[f'{time_prefix}{i + 1}' for i in range(mat.shape[1])])
        df = pd.concat([pd.DataFrame({id_col: ids}), values_df], axis=1)

    if output_path is not None:
        _save_tsv(df, output_path, index=False)
    return df
