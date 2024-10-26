
### Rhomboid Tiling Pooling
This repository contains code for performing Rhomboid Tiling Pooling on graph data, specifically designed for graph-level classification on Compound and Molecular datasets.

### Environment Setup
## Prerequisites
To run this project, ensure the following libraries are installed:

-argparse
-torch
-numpy
-torch_geometric
-tqdm
-rdkit

## Usage
# Data Preparation

For Compound datasets: Run get_rhomboid.py to preprocess data for Rhomboid Tiling Pooling.
For Molecular datasets: Run get_rhomboid_fromSMILES.py to generate the necessary information from SMILES data.
For both datasets, after initial preprocessing, run Dataset_RT_pooling.py to complete data preparation for rhomboid tiling pooling. This part can be automated using the script create_data.sh.

# Running the Main Program

To perform Rhomboid Tiling Pooling and graph-level classification:

For Compound datasets: Execute RTpool_comp.sh.
For Molecular datasets: Run RTpool_mol.sh.
These scripts will apply Rhomboid Tiling Pooling to the respective datasets and perform graph-level classification.

# Repository Structure
get_rhomboid.py - Preprocessing script for Compound datasets.
get_rhomboid_fromSMILES.py - Preprocessing for Molecular datasets from SMILES data.
Dataset_RT_pooling.py - Generates necessary information for rhomboid tiling pooling.
RTpool_comp.sh - Runs the main program on Compound datasets.
RTpool_mol.sh - Runs the main program on Molecular datasets.
create_data.sh - Automates data preparation steps.

# Additional information
Molecular/mol_graph_to_SMILES/ - Contains multiple scripts for converting molecular graphs to SMILES strings, enabling coordinate generation.
