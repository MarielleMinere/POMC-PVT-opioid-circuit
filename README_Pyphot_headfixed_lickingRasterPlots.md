# POMC-PVT-opioid-circuit
Python scripts for fiber photometry analysis in 'Thalamic opioids from POMC satiety neurons switch on sugar appetite'

# Pyphot_headfixed_lickingRasterPlots - Mouse Licking Behavior Analysis Script

## Overview
This script is designed to analyze the licking behavior of mice during various task conditions. It focuses on processing data from behavioral experiments where outcomes like water (H2O), sucrose, and no reward are tested. The script handles loading, processing, and plotting data to visualize and analyze the timing and frequency of licks.

## Requirements
To run this script, you'll need the following Python libraries installed:
- NumPy
- SciPy
- Matplotlib
- Plotly

## Files and Directories
- **MonkeyLogic Data**: The script expects `.mat` files containing trial and behavioral event data from MonkeyLogic tasks.
- **Output Directory**: All figures and processed data will be saved to the specified output directory.

### Main File Paths
- `PATH_ML`: Path to the MonkeyLogic data file.
- `output_folder`: Directory where output files will be saved.

## Usage
1. **Setting Paths**: Before running the script, ensure that the paths for the MonkeyLogic data and output folder are correctly set to match your local system configuration.
2. **Running the Script**: Execute the script in a Python environment where all required libraries are installed. The script will automatically process the data and generate output figures in the specified output directory.

## Functions
- **cues_time()**: Extracts and calculates the timing of cues and rewards from trial data.
- **calculate_licks()**: Processes lick data from trials, aligning and downsampling as necessary.
- **reduce_hz()**: Reduces the sampling rate of lick data to a specified final Hertz (Hz).
- **load_mat_file()**: Loads behavioral and trial data from a `.mat` file.
- **prossesing_licks()**: Prepares lick data for analysis by adding pre-trial lengths and cleaning up data.

## Output
- The script generates histograms of inter-lick intervals and raster plots of licking behavior across trials.

## Customization
You can adjust several parameters including sampling rates and event codes according to your experimental setup. These parameters are defined at the beginning of the script and can be modified as needed.
