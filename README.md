# Bernoulli Lookahead LSE

## To run the experiments
- Install the prerequisites (note that AEPsych requires a number of additional dependencies, including gpytorch and botorch). 
- Run `python run_experiments.py --help` to undersatnd the arguments. The defaults were used for the paper, on a c6l.metal node on EC2. 
- Run `python init_sensitivity_study.py` to run the experiment on sensitivity to number of initial Sobol points. 
- Run `python thresh_sensitivity_study.py` to run the experiment on sensitivity to LSE target. 
- Run `sh gentime_bench.sh` to run the timing run. Note that this runs the `gentime_bench.py` script 4 times with different args, since torch crashed if we tried to change the number of threads multiple times in a single script. 

## To make the figures. 
- Run script X to generate figure Y 
... 

## Human experiment

`human_data_collection` contains the [Psychopy](http://psychopy.org/) code needing to collect data for the real-world experiment in the paper. The experiment measures contrast sensitivity as a function of 6 variables: 

- spatial frequency
- temporal frequency
- mean luminance
- eccentricity
- field angle
- orientation
