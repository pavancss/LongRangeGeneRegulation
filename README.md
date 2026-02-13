# embryo_pipe
embryo_pipe is a specialized image analysis pipeline designed to automate the tracking of nuclei and the segmentation of transcription foci (MS2 and PP7) in developing embryos.

## Installation
To get up and running, follow these steps to set up your environment and install the package in editable mode.

### 1. Create the Conda Environment
Use the provided `.yml` file to install all necessary dependencies, including cellpose, scikit-image, and czifile.
```bash
conda env create -f imageanalysisEnv.yml
conda activate imageanalysis
```

### 2. Install the Package
Once the environment is active, install the package locally so that any changes you make to the source code are reflected immediately.
```bash
# Navigate to the root directory containing setup.py
pip install -e .
```

## Usage
The best way to explore the pipeline is via the provided `Test.ipynb` notebook. This notebook demonstrates the end-to-end workflow, from loading raw data to generating final intensity plots.

## The Core Pipeline: `process_embryo`
The `process_embryo` function automates the full analysis for a single embryo dataset. It integrates several modules from the package to transform raw microscopy stacks into quantified metrics.

### Pipeline Stages

**Data Import & Pre-processing:** Uses `import_image_czi` to load Zeiss `.czi` (or `.tif`) files, concatenating them into a temporal stack and generating Maximum Intensity Projections (MIPs) for MS2 and PP7 channels. 
- Modify this as per your imaging files and microscope file types. You may need to install nd2 image reader libraries etc.

**Nuclear Segmentation & Tracking:**
- Segments nuclei using a pre-trained Cellpose model.
- Tracks nuclei across time frames using a temporal overlap fraction.
- Filters out nuclei that overlap with the image border or exist for fewer than `surv_thresh` frames to ensure data quality.

**Foci Segmentation:**
- **MS2 (Green):** Uses Otsu thresholding and signal-to-noise ratios to identify transcription foci, filtering out small objects (noise) below a set pixel size.
- **PP7 (Red):** Segments foci specifically within the masks of tracked nuclei using a signal-to-noise ratio threshold (`r_cutoff`).

**Intensity Extraction:** Maps segmented foci back to individual nuclear labels over the entire time course.

**Normalization & Visualization:**
- Normalizes nuclear counts and calculates cumulative/instantaneous expression metrics.
- Generates a four-panel diagnostic plot showing MS2/PP7 expression and median foci intensities over time.
