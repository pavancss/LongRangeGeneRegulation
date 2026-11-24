import os
import numpy as np
import czifile
import re
from skimage.io import imread

def import_image_czi(imagefolder, ms2_channel=2, pp7_channel=1, imageorder=None):
    """
    Load and concatenate all movies of one embryo and max-project two channels.

    Every .czi (Zeiss) or .tif file in ``imagefolder`` is read, the files are
    concatenated along time, and the maximum intensity projection along Z is
    returned for the MS2 and PP7 channels.

    Inputs
    imagefolder- Folder holding the movies of a single embryo (hidden files
        starting with '.' are ignored).
    ms2_channel- Channel index (0-based) of the MS2 channel. Default 2.
    pp7_channel- Channel index (0-based) of the PP7 channel. Default 1.
    imageorder- Optional list of file names giving the time order. By default
        files are sorted by the first number in their name (e.g. emb_1.czi,
        emb_2.czi, emb_10.czi), so every file name must contain a number.

    Outputs
    G_max- (T, Y, X) max projection of the MS2 channel.
    R_max- (T, Y, X) max projection of the PP7 channel.

    Notes
    - After squeezing, .czi files are expected as (C, T, Z, Y, X); .tif files
      are expected with channels on axis 4 (moved to axis 0). A file with a
      single time point (4D after squeezing) gets a time axis added.
    - The returned stacks start at the first frame of the first file, so
      ``frame_start`` / ``frame_stop`` in the notebook index from there.
    """

    files = os.listdir(imagefolder)
    czi_files = [f for f in files if (f.endswith('.czi') or f.endswith('.tif')) and not f.startswith('.')]

    if not imageorder:
        imageorder = sorted(czi_files, key=lambda x: int(re.findall(r'\d+', x)[0]))

    im_concat = np.array([])
    for filename in imageorder:
        path = os.path.join(imagefolder, filename)
        if filename.endswith('.czi'):
            image = np.squeeze(czifile.imread(path))
        elif filename.endswith('.tif'):
            image = np.moveaxis(imread(path), 4, 0)
        
        if image.ndim < 5:
            image = np.expand_dims(image, axis=1)
            
        im_concat = image if im_concat.size == 0 else np.concatenate((im_concat, image), axis=1)

    G_max = np.max(im_concat[ms2_channel, ...], axis=1)
    R_max = np.max(im_concat[pp7_channel, ...], axis=1)
    return G_max, R_max