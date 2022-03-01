# Bernoulli Level Set Estimation 

This repository contains the code associated with the paper [Look-Ahead Acquisition Functions for Bernoulli Level Set Estimation]().

If you find this code useful, please cite it as

    @inproceedings{BernoulliLSE,
        author    = {Letham, Benjamin and Guan, Phillip and Tymms, Chase and Bakshy, Eytan and Shvartsman, Michael},        
        title     = {Look-Ahead Acquisition Functions for {B}ernoulli Level Set Estimation},
        booktitle   = {Proceedings of the 25th International Conference on Artificial Intelligence and Statistics},
        year      = {2022},
        series = {AISTATS},
    }

## The acquisition functions

The acquisition function implementations are Botorch acquisition classes, and are included in AEPsych, [here](https://github.com/facebookresearch/aepsych/blob/main/aepsych/acquisition/lookahead.py). Necessary functions for computing the look-ahead level set posterior are also in AEPsych, [here](https://github.com/facebookresearch/aepsych/blob/main/aepsych/acquisition/lookahead_utils.py).

## To run the experiments

- Install the latest main branch of the [AEPsych](https://github.com/facebookresearch/aepsych) package from github. It is best to grab the latest version, but at the least the version must be later than [this commit](https://github.com/facebookresearch/aepsych/commit/288b3265ddde3b864e5e369bee9eeff7c3204f76) in order to run all of the code in this repository. Note that AEPsych requires a number of additional dependencies, such as gpytorch and botorch, which can be installed via pip.
- Run `python run_experiments.py --help` to understand the arguments. This script will run the full suite of benchmarks behind Figs. 3, 5, S1, S3, and S4. The defaults given were used for the paper, on a c6l.metal node on EC2. 
- Run `python init_sensitivity_study.py` to run the experiment on sensitivity to number of initial Sobol points, as in Fig. S5.
- Run `python thresh_sensitivity_study.py` to run the experiment on sensitivity to LSE target, as in Fig. S6.
- Run `sh gentime_bench.sh` to run the timing run, used for Fig. S2. Note that this runs the `gentime_bench.py` script 4 times with different args, since torch crashed if we tried to change the number of threads multiple times in a single script. 

Results for each set of benchmarks will be placed in the `data/` subdirectory.

## To make the figures.

- Run `figures/plot_posteriors.py` to generate Fig. 1.
- Run `figures/plot_acquisition.py` to generate Fig. 2.
- Run `figures/plot_experiment_results.py` to generate Figs. 3 and 5.
- Run `figures/make_stim_plots.py` to generate Fig. 4. 
- Run `figures/plot_supplement_experiment_results.py` to generate Figs. S1 and S3.
- Run `figures/plot_gentimes.py` to generate Fig. S2.
- Run `figures/plot_edge_sampling.py` to generate Fig. S4.
- Run `figures/plot_init_sensitivity_results.py` to generate Fig. S5.
- Run `figures/plot_thresh_sensitivity_results.py` to generate Fig. S6.

## Human experiment

`human_data_collection` contains the [Psychopy](http://psychopy.org/) code needing to collect data for the real-world experiment in the paper. The experiment measures contrast sensitivity as a function of 6 variables: 

- pedestal (mean background luminance)
- contrast
- temporal frequency
- spatial frequency
- eccentricity
- size (visual field angle)
- orientation
