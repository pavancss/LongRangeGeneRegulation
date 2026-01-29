import os
import numpy as np
import czifile
import re
from skimage.io import imread

def import_image_czi(imagefolder, ms2_channel=2, pp7_channel=1, imageorder=[]):
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