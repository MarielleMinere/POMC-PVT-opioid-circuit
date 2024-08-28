# Doric Refeeding-Refed Freely moving 35 min analysis script

## Overview
This script is designed to analyze the changes in dF/F compared to baseline during either refeeding or in the refed state. It focuses on calculating the dF/F of .doric photometry data. The script handles loading, processing, plotting and exporting data in .csv files.

## Requirements
To run this script, you'll need the following Python libraries installed:
- NumPy
- SciPy
- Matplotlib
- Plotly

## Files and Directories
- **Doric Data**: The script expects .doric files in the PATH variable 
- **Output Directory**: All output files will be saved to the PATH directory.

### Main File Paths
- `PATH`: Path for the Doric datafile

## Usage
1. **Setting Path**: Before running the script, ensure that the path for the Doric data is correctly set to match your local system configuration.
2. **Running the Script**: Execute the script in a Python environment where all required libraries are installed. The script will automatically process the data and generate output figures in the specified output directory.

## Functions
- **get_photometry_data()**: Extracts the isosbestic and gcamp signal from the doric file as analog1 and analog2
- **scaling factor()**: scales the isosbestic signal (analog2) to the same amplitude as the gcamp signal (analog1)
- **df()**: calculates the dF/F using the baseline as F0
- **butter_lowpass_filter()**: Puts a lowpass filter on the signal
- **linegraph()**: Visualises the resultant trace and saves the graph as a png in the PATH directory

## Output
- The script generates a visual plot of the dF/F and saves a .csv file with the data

## Customization
You can adjust several parameters including sampling rates (SEC_HZ) and the baseline period (BASE_LENGTH). These parameters are defined at the beginning of the script and can be modified as needed.
