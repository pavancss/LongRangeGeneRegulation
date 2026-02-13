import numpy as np
from cellpose import models
from skimage.measure import label
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import dilation, remove_small_objects
from skimage.segmentation import relabel_sequential


def nonzeroavg(img):
    """Calculates the average of non-zero pixels."""
    return np.mean(img[np.nonzero(img)])


def pp7_segmentation(R_analyze, tracked_nucs_filter, div_thresh):
    """
    Segments PP7 transcription foci based on a signal-to-noise ratio threshold.
    
    Args:
        R_analyze (np.array): 3D stack (T, Y, X) of the Red/PP7 channel.
        tracked_nucs_filter (np.array): Mask of tracked nuclei. Needs to only be binary
        div_thresh (float): Intensity ratio threshold for foci detection.
        
    Returns:
        R_foci mask- Labeled mask of PP7 foci inside nuclei 
        R_foci_int- Backround subtracted intensity image time course of all PP7 foci. 
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
        R_div[R_div == np.nan] = 0
        foci_mask = (R_div > div_thresh) * (tracked_nucs_filter[timectr, :, :] > 0)
        
        foci_label = label(foci_mask)
        R_foci[timectr, :, :] = foci_label
        
        # Background subtraction
        bgmask = (tracked_nucs_filter[timectr, :, :] > 0) & (~foci_mask)
        bgval = nonzeroavg(bgmask * R_analyze[timectr, :, :])
        int_image = (R_analyze[timectr, :, :] - np.full(bgmask.shape, fill_value=bgval)) * foci_mask
        R_foci_int[timectr, :, :] = int_image
        
    return R_foci, R_foci_int


def ms2_segmentation(G_analyze, tracked_nucs_filter, div_thresh,min_size=2):
    """
    Segments MS2 transcription foci based on a signal-to-noise ratio threshold.
    
    Args:
        G_analyze (np.array): 3D stack (T, Y, X) of the Green/MS2 channel.
        tracked_nucs_filter (np.array): Mask of tracked nuclei. Needs to only be binary
        div_thresh (float): Intensity ratio threshold for foci detection.
        min_size (integer) - Minimum MS2 focus size (in pixels)
        
    Returns:
        G_foci mask- Labeled mask of PP7 foci inside nuclei 
        G_foci_int- Backround subtracted intensity image time course of all PP7 foci. 
    """
    G_foci = np.zeros(G_analyze.shape)
    G_foci_int = np.zeros(G_analyze.shape)

    for timectr in range(G_analyze.shape[0]):
        im = G_analyze[timectr, :, :].astype('float')
        threshs = threshold_otsu(im)
        foci_mask = im > threshs
        
        bg = np.mean(im[~foci_mask])
        foci = im * foci_mask
        foci[foci == 0] = np.nan
        
        G_div = foci / bg
        foci_mask_clear = (G_div > div_thresh) * (tracked_nucs_filter[timectr, :, :] > 0)
        
        foci_label = label(foci_mask_clear)
        foci_label = remove_small_objects(foci_label, min_size=2)
        foci_label_filt, fw, rv = relabel_sequential(foci_label)
        
        G_foci[timectr, :, :] = foci_label_filt
        
        # Background subtraction
        bgmask = ~foci_mask_clear
        bgval = nonzeroavg(bgmask * G_analyze[timectr, :, :])
        int_image = (G_analyze[timectr, :, :] - np.full(bgmask.shape, fill_value=bgval)) * foci_mask_clear
       
        G_foci_int[timectr, :, :] = int_image

    return G_foci, G_foci_int


def nuc_overlap(singlenucmask, allnucmask, overlap_frac=0.9):
    """
    Function to find out nucleus label in next time frame of movie using overlap_frac cutoff

    Inputs
    singlenucmask- Single nuclear binary mask in frame T. 2D numpy array. 
    allnucmask- Labeled nuclear mask of frame T+1 in which to find nucleus from singlenucmask. 2D numpy array 
    overlap_frac- Optional. Nucleus is counted as same if there is at least 90% overlap. 

    Output
    Label of nucleus in frame T+1. If no nucleus shows >overlap_frac overlap, then -1. 

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
        

def nuclei_track(R_analyze, overlap_frac=0.5, cellposemodel='nuclei'):
    """
    Function to segment and track nuclei 

    Inputs
    R_analyze- T x X x Y stack where the nuclei are tracked in time axis T.
    overlap_frac- Optional. Minimum overlap between nuclei in successive frames. 
    cellposemodel- This uses models in cellpose v3.1.0. Change the base path (Does not work well with standard cellpose v4 models)

    Outputs

    nucs_commonlabels- Nuclear label image T x X x Y where the nuclei are tracked over time with same label
    nuc_counts - List of number of nuclei in each frame. Useful to track for segmentation quality.
    nuc_masks_filter - Labeled nuclear masks where each frame is segmented and labeled individually


    """
    channels = [[0, 0]]
    
    # Model Selection
    base_path = 'cellpose_models/'
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
    Function to filter the nuclei for overlap with the border and for nuclei that exist in frame for a short time

    Inputs
    tracked nucs- Nuclear label image T x X x Y where the nuclei are tracked over time with same label
    surv_thresh- In pixels, the width of the border band. If nuclei overlap this band, they are removed. 

    Outputs

    nuc_survival- List of number of frames each label in tracked_nucs_filter survives in the whole movie.
    tracked_nucs_filter- Filtered nuclear label image T x X x Y where the nuclei are tracked over time with same label

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