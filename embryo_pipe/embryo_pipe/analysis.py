import numpy as np
from skimage.measure import label
import pandas as pd
import os

def singlenuc_ints(tracked_nucs_filter, G_foci_int=np.zeros((1,1)), R_foci_int=np.zeros((1,1))):
    ''' 
    This function takes an input of nuclear masks tracked through time and the MS2 and PP7 segmentation  and
    outputs the single nucleus intensities over time. 

    Inputs- tracked_nucs_filter-> labeled nuclear mask images of nuclei tracked through time.
    G_foci -> MS2 imaging foci masks. Need not be labeled for the purposes of this function
    R_foci -> PP7 imaging foci masks. 


    Outputs
    gburstInt_singlenuc, rburstInt_singlenuc-> Dictionaries where the keys are nuclear mask labels and the values are lists of intensities of foci over the course of a movie.
    If nucleus is not detected at a time point, then the value is stored as np.nan. If nucleus is called, but there is no foci, value is 0. 

    '''

    gburstInt_singlenuc={}
    rburstInt_singlenuc={}

    if not np.array_equal(G_foci_int,np.zeros((1,1))):
        for nuclabel in range(1,np.max(tracked_nucs_filter)):
            nuc_mask=tracked_nucs_filter==nuclabel
            foci_g=G_foci_int*nuc_mask
            gburstInt_singlenuc[nuclabel]=[np.sum(foci_g[timectr,:,:]) for timectr in range(tracked_nucs_filter.shape[0])]
    
    if not np.array_equal(R_foci_int,np.zeros((1,1))):
        for nuclabel in range(1,np.max(tracked_nucs_filter)):
            nuc_mask=tracked_nucs_filter==nuclabel
            foci_r=R_foci_int*nuc_mask
            rburstInt_singlenuc[nuclabel]=[np.sum(foci_r[timectr,:,:]) for timectr in range(tracked_nucs_filter.shape[0])]

    if (len(gburstInt_singlenuc)>0) & (len(rburstInt_singlenuc)>0):
        return gburstInt_singlenuc,rburstInt_singlenuc
    elif len(gburstInt_singlenuc)>0 :
        return rburstInt_singlenuc
    elif len(rburstInt_singlenuc)>0 :
        return gburstInt_singlenuc
    else:
        print('Provide non-empty label masks for foci')
        return np.array([]),np.array([])


def normNucCount(gburstInt_singlenuc,rburstInt_singlenuc):
    ''' 
    This function takes MS2 and PP7 foci single nuclear intensity dictionaries and outputs normalized data

    Inputs- 
    gburstInt_singlenuc, rburstInt_singlenuc-> Dictionaries where the keys are nuclear mask labels and the values are lists of intensities of foci over the course of a movie. 
    If nucleus is not detected at a time point, then the value is stored as np.nan. If nucleus is called, but there is no foci, value is 0. 

    Outputs-
    normed_tcourse- Number of nuclei (with MS2 expression any point in the movie) with a burst at given time in R (PP7 channel), 
    cum_tcourse- Cumulative of normed_tcourse.
    scyl_instexp- Sum of intensities of all bursts at any given time in gbursts (MS2 channel)
    chrb_instexp- Sum of intensities of all bursts at any given time in gbursts (PP7 channel). But analyzed only in nuclei that show MS2 signal at any point in the movie. 
    scyl_fociInt- gbursts median intensity of burst (at any given time, not total burst size)
    chrb_fociInt- rbursts median intensity of burst (at any given time, not total burst size)

    
    '''

    scyl_pos_nucs={}
    chrb_pos_nucs={}
    for nuc in gburstInt_singlenuc.keys():
        g_ints=np.array(gburstInt_singlenuc[nuc])
        np.nan_to_num(g_ints,copy=False)

        r_ints=np.array(rburstInt_singlenuc[nuc])
        np.nan_to_num(r_ints,copy=False)
        if np.sum(g_ints)>0:
            scyl_pos_nucs[nuc]=g_ints
            chrb_pos_nucs[nuc]=r_ints
    tot_scyl_posnucs=len(scyl_pos_nucs.keys())
    r_ints_mat=np.zeros((tot_scyl_posnucs,len(gburstInt_singlenuc[1])))
    g_ints_mat=np.zeros((tot_scyl_posnucs,len(gburstInt_singlenuc[1])))
    ctr=0
    for nuc in scyl_pos_nucs.keys():
        r_ints_mat[ctr,:]=chrb_pos_nucs[nuc]
        g_ints_mat[ctr,:]=scyl_pos_nucs[nuc]
        ctr+=1
    
    pos_nucs=r_ints_mat>0
    tcourse=np.sum(pos_nucs,axis=0)
    normed_tcourse=100*(tcourse/tot_scyl_posnucs)
    cum_tcourse=[]

    scyl_instexp=[]
    chrb_instexp=[]
    scyl_fociInt=[]
    chrb_fociInt=[]

    for time in range(len(gburstInt_singlenuc[1])):
        col_sum=np.sum(pos_nucs[:,0:time],axis=1)
        scyl_time=g_ints_mat[:,time]
        chrb_time=r_ints_mat[:,time]
        scyl_instexp.append(np.sum(scyl_time))
        scyl_fociInt.append(np.median(scyl_time[np.nonzero(scyl_time)]))
        chrb_instexp.append(np.sum(chrb_time))
        chrb_fociInt.append(np.median(chrb_time[np.nonzero(chrb_time)]))

        cum_tcourse.append(100*(np.sum(col_sum>0)/tot_scyl_posnucs))
    frac_scyl_posnucs=tot_scyl_posnucs/max(gburstInt_singlenuc.keys())

    return normed_tcourse,cum_tcourse,scyl_instexp,chrb_instexp, scyl_fociInt, chrb_fociInt



def burst_metrics(burstInt_singlenuc):
    '''
    This function analyzes time series of burst intensities in a single nuclei for an entire image to generate a list of transcription 
    ON times, OFF times, and total burst intensity during each ON state. 

    Input-> Is a dictionary where the values to each key is a list of foci intensities in the nucleus specified by the key. 
    The key is canonically the nucleus label, but the function doesn't use that fact. 

    Output-> ON length list (unsorted). Just a sequence of ON state times as observed in the loop.
    OFF length list. Also unsorted. The list ignores OFF states that start from the beginning of the movie or continue till the end. 
    Burst intensity list- Unsorted. Sequence of total burst intensity (summed). Should correspong to ON length list.


    '''
    burst_lens=[]
    burst_sizes=[]
    pause_lens=[]

    for nucctr in range(1,max(burstInt_singlenuc.keys())):

        bursts=np.array(burstInt_singlenuc[nucctr])>0
        #bursts=bursts[:-5]
        burst_label=label(bursts)
        
        for lab in range(1,np.max(burst_label)+1):
            burst=burst_label==lab
            if burst[-1]!=1:
                int_list=np.array(burstInt_singlenuc[nucctr])
                #int_list=int_list[:-5]
                burst_lens.append(np.sum(burst))
                burst_sizes.append(np.sum(int_list[burst]))

        #pauses=np.array(burstInt_singlenuc[nucctr])==0
        pauses=~bursts
        pause_label=label(pauses)

        for lab in range(1,np.max(pause_label)+1):
            pause=pause_label==lab
            if (pause[0]==1) | (pause[-1]==1):
                continue
            
            pause_lens.append(np.sum(pause))

    
    return burst_lens, pause_lens, burst_sizes

def positive_nucs(tracked_nucs_filter,foci_mask):

    pos_nucs={}
    for timectr in range(tracked_nucs_filter.shape[0]):
        nucs=tracked_nucs_filter[timectr,:,:]
        pos_nucs[timectr]=np.unique(nucs*(foci_mask[timectr]>0))

    num_pos_nucs=[len(pos_nucs[ctr]) for ctr in range(len(pos_nucs))]

    return pos_nucs,num_pos_nucs


def rburst_metrics(burstInt_singlenuc,gburstInt_singlenuc):
    '''
    This function analyzes time series of burst intensities in a single nuclei for an entire image to generate a list of transcription 
    ON times, OFF times, and total burst intensity during each ON state. 

    Input-> Is a dictionary where the values to each key is a list of foci intensities in the nucleus specified by the key. 
    The key is canonically the nucleus label, but the function doesn't use that fact. 

    Output-> ON length list (unsorted). Just a sequence of ON state times as observed in the loop.
    OFF length list. Also unsorted. The list ignores OFF states that start from the beginning of the movie or continue till the end. 
    Burst intensity list- Unsorted. Sequence of total burst intensity (summed). Should correspong to ON length list.


    '''
    burst_lens=[]
    burst_sizes=[]
    pause_lens=[]

    for nucctr in range(1,max(burstInt_singlenuc.keys())):

        g_ints=np.array(gburstInt_singlenuc[nucctr])
        np.nan_to_num(g_ints,copy=False)

        if np.sum(g_ints)>0:


            bursts=np.array(burstInt_singlenuc[nucctr])>0
            #bursts=bursts[:-5]
            burst_label=label(bursts)
            
            for lab in range(1,np.max(burst_label)+1):
                burst=burst_label==lab
                if burst[-1]!=1:
                    int_list=np.array(burstInt_singlenuc[nucctr])
                    #int_list=int_list[:-5]
                    burst_lens.append(np.sum(burst))
                    burst_sizes.append(np.sum(int_list[burst]))

            #pauses=np.array(burstInt_singlenuc[nucctr])==0
            pauses=~bursts
            pause_label=label(pauses)

            for lab in range(1,np.max(pause_label)+1):
                pause=pause_label==lab
                if (pause[0]==1) | (pause[-1]==1):
                    continue
                
                pause_lens.append(np.sum(pause))

    
    return burst_lens, pause_lens, burst_sizes



def create_timeseries_dataframe(
    list_of_lists,
    list_identifiers,
    time_per_image,
    total_time=0,
    output_path=None,
    column_prefix='emb',
    min_count=3
):
    """
    Create a time-series dataframe from a list of lists with mean and SEM calculations.
    
    Parameters:
    -----------
    list_of_lists : list of lists
        Each inner list contains time-series data for one identifier
    list_identifiers : list
        Identifiers for each list (e.g., embryo IDs)
    time_per_image : float
        Time interval between measurements
    total_time : float, default=60
        Total time span for the experiment
    output_path : str, optional
        Full path (including filename) where CSV will be saved
    column_prefix : str, default='emb'
        Prefix for column names (e.g., 'normed_ctrlemb', 'cum_ctrlemb')
    min_count : int, default=3
        Minimum number of non-NaN values required to calculate mean/SEM
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with Time column, individual data columns, mean and SEM columns
    """
    
    # Compute max length
    max_len = max(len(lst) for lst in list_of_lists)
    print(max_len)
    
    # Padding function
    def pad_list(lst, length):
        return [np.nan] * (length - len(lst)) + lst
    
    # Pad all lists
    padded_lists = [pad_list(lst, max_len) for lst in list_of_lists]
    
    # Create column names
    colnames = [f'{column_prefix}{x}' for x in list_identifiers]
    
    # Build DataFrame
    df = pd.DataFrame(dict(zip(colnames, padded_lists)))
    
    # Add Time column
    df["Time"] = [total_time - time_per_image * (len(df) - ctr) for ctr in range(len(df))]
    
    # Function to calculate mean and SEM with minimum count requirement
    def mean_sem_with_min_count(df_inner, cols, min_n):
        count = df_inner[cols].count(axis=1)
        mean = df_inner[cols].mean(axis=1, skipna=True)
        sem = df_inner[cols].std(axis=1, ddof=1, skipna=True) / count**0.5
        mean[count < min_n] = np.nan
        sem[count < min_n] = np.nan
        return mean, sem
    
    # Compute mean and SEM
    df["mean"], df["sem"] = mean_sem_with_min_count(df, colnames, min_count)
    
    # Reorder columns
    df = df[["Time"] + colnames + ["mean", "sem"]]
    
    # Drop rows where ctrl_mean is NaN
    df = df.dropna(subset=["mean"])
    
    # Reset index
    df.reset_index(inplace=True, drop=True)
    
    # Save CSV if output path is provided
    if output_path is not None:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df.to_csv(output_path, sep='\t', index=False)
    
    return df
