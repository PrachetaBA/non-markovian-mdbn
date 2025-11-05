## Overview

This branch is used for all experiments related to Markovian queuing network example as described in the journal paper. The general workflow is to first generate training data using the specified event-based simulator `simulator.py`. Next, we extract 2-TBN style and time series style data using the scripts `subsampling_2tbn.py` and `subsampling_dbn.py`. We construct the corresponding DBN using `construct_dbn.py` and run inference using `inference.py`. Add the details of the experiment in the evidence file under `data/evidence`. Generate the ground truth data using `simulator_interventions.py`. Finally we use `utils/compare_qnetwork_dbn.py` to generate the figures and comparison plots. 

The detailed instructions to run these scripts and other experiments is specified [here](/README_detailed.md). 

## Local configuration
Used conda environment mdbn-pyagrum, and the corresponding `environment.yml` file.
