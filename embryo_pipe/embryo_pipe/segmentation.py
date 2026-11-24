import os
import numpy as np
from cellpose import models
from skimage.measure import label
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects
from skimage.segmentation import relabel_sequential


here = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(here, 'cellpose_models')


def nonzeroavg(img):
    """
    Mean of the non-zero pixels of an array.

    Used for background estimation on masked images, where pixels outside the
    mask are 0. Returns NaN (with a RuntimeWarning) if every pixel is 0.
    """
    return np.mean(img[np.nonzero(img)])


def pp7_segmentation(R_analyze, tracked_nucs_filter, div_thresh):
    """
    Segment PP7 transcription foci frame by frame with a signal-to-noise threshold.

    For each frame, an Otsu threshold separates nuclear signal from background;
    each pixel is divided by the mean nuclear intensity, and pixels above
    ``div_thresh`` that fall inside a tracked nucleus are called foci.

    Args:
        R_analyze (np.ndarray): (T, Y, X) stack of the PP7 channel (max projection).
        tracked_nucs_filter (np.ndarray): (T, Y, X) nuclear masks. Any value > 0
            counts as nucleus, so a labeled or binary stack both work.
        div_thresh (float): Pixel / mean-nuclear-intensity ratio above which a
            pixel is called a focus.

    Returns:
        R_foci (np.ndarray): (T, Y, X) labeled focus mask (labels restart in
            every frame).
        R_foci_int (np.ndarray): (T, Y, X) focus intensities after subtracting
            the mean non-focus nuclear intensity of that frame; 0 outside foci.
    """
    R_foci = np.zeros(R_analyze.shape)
    R_foci_int = np.zeros(R_analyze.shape)

    for timectr in range(R_analyze.shape[0]):
        im = R_analyze[timectr, :, :].astype('float')
        threshs = threshold_otsu(im)
        cell_mask = im > threshs
        
        cells = im * cell_mask
        cells[cells == 0] = np.nan
        im[~cell_mask] = np.nan
        
        # Calculate ratio and mask foci within tracked nuclei
        R_div = im / (cell_mask * np.mean(cells[cells > 0]))
        R_div[np.isnan(R_div)] = 0
        foci_mask = (R_div > div_thresh) * (tracked_nucs_filter[timectr, :, :] > 0)
        
        foci_label = label(foci_mask)
        R_foci[timectr, :, :] = foci_label
        
        # Background subtraction
        bgmask = (tracked_nucs_filter[timectr, :, :] > 0) & (~foci_mask)
        bgval = nonzeroavg(bgmask * R_analyze[timectr, :, :])
        int_image = (R_analyze[timectr, :, :] - np.full(bgmask.shape, fill_value=bgval)) * foci_mask
        R_foci_int[timectr, :, :] = int_image
        
    return R_foci, R_foci_int


def ms2_segmentation(G_analyze, tracked_nucs_filter, div_thresh=1.25,min_size=2):
    """
    Segment MS2 transcription foci with a signal-to-noise threshold.

    The whole stack is smoothed (Gaussian, sigma=1), divided by the mean smoothed
    intensity inside nuclei (one value for the whole movie), thresholded at
    ``div_thresh``, and objects smaller than ``min_size`` are removed.

    Args:
        G_analyze (np.ndarray): (T, Y, X) stack of the MS2 channel (max projection).
        tracked_nucs_filter (np.ndarray): (T, Y, X) nuclear masks. Any value > 0
            counts as nucleus.
        div_thresh (float): Smoothed-intensity / mean-nuclear-intensity ratio
            above which a pixel is called a focus.
        min_size (int): Minimum focus size in pixels. Note that
            ``remove_small_objects`` sees the full (T, Y, X) stack, so
            connectivity is 3D (a focus can connect across adjacent frames).

    Returns:
        G_foci (np.ndarray): (T, Y, X) boolean focus mask (not labeled).
        G_foci_int (np.ndarray): (T, Y, X) raw intensity minus the movie-wide
            mean nuclear background, 0 outside foci.
    """
    G_foci = np.zeros(G_analyze.shape)
    G_foci_int = np.zeros(G_analyze.shape)

    im = G_analyze.astype('float')
    im_smooth = gaussian(im, sigma=1)
    nucs=tracked_nucs_filter>0

    bg = np.mean(im_smooth[nucs])
    emb = im_smooth * nucs

    G_div = emb / bg
    #g_bin=G_div > threshold_multiotsu(G_div[np.nonzero(G_div)], classes=4)[2]
    g_bin=G_div > div_thresh
    #g_bin_erode=isotropic_opening(g_bin,radius=1)
    G_foci=remove_small_objects(g_bin,min_size=min_size)
    G_foci_int = (G_analyze - np.full(G_analyze.shape, fill_value=bg)) * G_foci

    return G_foci, G_foci_int


def nuc_overlap(singlenucmask, allnucmask, overlap_frac=0.9):
    """
    Find which labeled nucleus in another frame matches a single nucleus.

    The label with the most overlapping pixels is chosen; it is accepted only if
    the overlap exceeds ``overlap_frac`` of that candidate's area.

    Inputs
    singlenucmask- 2D binary mask of one nucleus.
    allnucmask- 2D labeled nuclear mask of the frame to search in
        (``nuclei_track`` passes the PREVIOUS, already-tracked frame).
    overlap_frac- Optional, default 0.9. Minimum overlap as a fraction of the
        candidate nucleus area.

    Output
    Matching label in ``allnucmask``, or -1 if none passes the cutoff.

    """
    overlap = (singlenucmask > 0) * allnucmask
    labels, counts = np.unique(overlap[np.nonzero(overlap)], return_counts=True)

    if len(labels) == 0:
        return -1 
    else:
        counts_max = counts[np.argmax(counts)]
        overlap_label = labels[np.argmax(counts)]
        overlap_mask = allnucmask == overlap_label
        
        if counts_max > overlap_frac * np.sum(overlap_mask):
            return overlap_label
        else:
            return -1
        

def nuclei_track(R_analyze, overlap_frac=0.5, cellposemodel='nuclei', model_dir=DEFAULT_MODEL_DIR):
    """
    Segment nuclei in every frame with Cellpose and link them over time.

    Bright pixels (> 10000, i.e. transcription foci) are replaced by the stack
    mean before segmentation. Each frame is smoothed (sigma=1.5) and segmented
    (diameter=30); objects < 100 px are removed. Nuclei are then linked
    frame-to-frame with ``nuc_overlap``; unmatched nuclei get a new label.

    Inputs
    R_analyze- (T, Y, X) stack used for nuclear segmentation (PP7 channel
        nuclear background).
    overlap_frac- Optional, default 0.5. Minimum overlap used to link nuclei in
        successive frames (see ``nuc_overlap``).
    cellposemodel- 'nuclei' (-> nucleitorch_0), 'cp' (-> CP) or the file name
        of another model in ``model_dir``. Written for cellpose v3.1.0; does not
        work well with standard cellpose v4 models.
    model_dir- Folder containing the Cellpose models.

    Outputs (in this order)
    nuc_masks_filter- (T, Y, X) per-frame labels (not linked over time).
    nuc_counts- List with the number of nuclei segmented in each frame, useful
        for checking segmentation quality.
    nucs_commonlabels- (T, Y, X) labels linked over time (same nucleus keeps
        the same label).

    Note: when linking, labels are iterated as ``range(1, max_label)`` so the
    highest-numbered nucleus in each frame (after frame 0) is not carried over.

    """
    channels = [[0, 0]]
    
    # Model Selection
    base_path = os.path.join(model_dir, '')
    if cellposemodel == 'nuclei':
        model_path = base_path + 'nucleitorch_0'
    elif cellposemodel == 'cp':
        model_path = base_path + 'CP'
    else:
        model_path = base_path + cellposemodel
        
    model = models.CellposeModel(gpu=False, pretrained_model=model_path)

    # Pre-processing to remove bright foci before nuclear segmentation
    simplefoci = R_analyze > 10000
    R_focidel = R_analyze.copy()
    R_focidel[simplefoci] = np.mean(R_analyze)
    
    nuc_masks_nt = np.zeros(R_analyze.shape)
    
    for ctr in range(R_analyze.shape[0]):
        tmp = gaussian(R_focidel[ctr, :, :], sigma=1.5)
        nuc_masks_nt[ctr, :, :], flows, styles = model.eval(tmp, diameter=30, channels=channels)

    nuc_masks_filter = np.zeros(nuc_masks_nt.shape)
    nuc_counts = []

    for ctr in range(nuc_masks_nt.shape[0]):
        filt = remove_small_objects(np.uint32(nuc_masks_nt[ctr, :, :]), min_size=100)
        nuc_masks_filter[ctr, :, :], fw, rv = relabel_sequential(filt)
        nuc_counts.append(int(np.max(nuc_masks_filter[ctr, :, :])))

    # Sequential Tracking
    nucs_commonlabels = np.zeros(nuc_masks_filter.shape)
    nucs_commonlabels[0, :, :] = nuc_masks_filter[0, :, :]
    
    for timectr in range(1,R_analyze.shape[0]):
        curr_slice=nuc_masks_filter[timectr,:,:]
        curr_slice_update=np.zeros(curr_slice.shape)
        prev_slice=nucs_commonlabels[timectr-1,:,:]
        max_nucs=np.max(nucs_commonlabels)
        for nuc_label in range(1,int(np.max(curr_slice))):
            singlenucmask=curr_slice==nuc_label
            overlap_label=nuc_overlap(singlenucmask,prev_slice,overlap_frac)
            if overlap_label==-1:
                curr_slice_update[singlenucmask]=max_nucs+1
                max_nucs+=1
            else:
                curr_slice_update[singlenucmask]=overlap_label
        nucs_commonlabels[timectr,:,:]=curr_slice_update

    return nuc_masks_filter, nuc_counts, nucs_commonlabels
        
def nucs_border_time_filter(tracked_nucs,surv_thresh=5):

    """
    Remove tracked nuclei that touch the image border or are present only briefly.

    A nucleus is removed if it overlaps the image edge in ANY frame (the edge band
    is 1 px at the top/left and 2 px at the bottom/right), or if it is present
    in ``surv_thresh`` frames or fewer. Kept nuclei are relabeled 1..N.

    Inputs
    tracked_nucs- (T, Y, X) labels linked over time (``nuclei_track`` output).
        Modified in place: border-touching nuclei are set to 0.
    surv_thresh- Minimum number of frames, exclusive (nucleus must be present
        in more than ``surv_thresh`` frames to be kept). Default 5.

    Outputs
    nuc_survival- Dict: original label -> number of frames present, for all
        non-border nuclei (including ones removed for short survival).
    tracked_nucs_filter- (T, Y, X) uint32 filtered and relabeled stack.

    """

    nuc_survival={}
    innermask=np.zeros(tracked_nucs.shape[1:])
    innermask[1:-2,1:-2]=1
    bordermask=~(innermask>0)
    tracked_nucs_filter=np.zeros(tracked_nucs.shape)
    filtered_ctr=1
    for nuc in range(1,int(np.max(tracked_nucs))):
        nuc_mask=tracked_nucs==nuc
        if (np.sum(nuc_mask*bordermask)>0):
            tracked_nucs[nuc_mask]=0
        else:
            survival=[1 for ctr in range(tracked_nucs.shape[0]) if np.max(nuc_mask[ctr,:,:])>0]
            nuc_survival[nuc]=np.sum(survival)
            if nuc_survival[nuc]>surv_thresh:
                tracked_nucs_filter[nuc_mask]=filtered_ctr
                filtered_ctr+=1
    tracked_nucs_filter=np.uint32(tracked_nucs_filter)
    return nuc_survival,tracked_nucs_filter