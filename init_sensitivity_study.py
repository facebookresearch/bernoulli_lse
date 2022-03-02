# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import os
import argparse

# run each job single-threaded, paralellize using pathos
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# multi-socket friendly args
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["KMP_BLOCKTIME"] = "1"
import torch

# force torch to 1 thread too just in case
torch.set_num_interop_threads(1)
torch.set_num_threads(1)

import time
from copy import deepcopy
from pathlib import Path

from aepsych.benchmark import BenchmarkLogger, PathosBenchmark, combine_benchmarks

from problems import (
    DiscrimHighDim,
    Hartmann6Binary,
    ContrastSensitivity6d,  # This takes a few minutes to instantiate due to fitting the model
)

chunks = 5
reps_per_chunk = 20
log_frequency = 10
large_opt_size = 750
nproc = 124
global_seed = 1000
inits = [100, 250, 500]

if __name__ == "__main__":

    out_fname_base = Path("../data/init_sensitivity")
    out_fname_base.mkdir(
        parents=True, exist_ok=True
    )  # make an output folder if not exist
    problems = [
        DiscrimHighDim(),
        Hartmann6Binary(),
        ContrastSensitivity6d(),
    ]
    bench_config = {
        "common": {
            "outcome_type": "single_probit",
            "strategy_names": "[init_strat, opt_strat]",
        },
        "init_strat": {"generator": "SobolGenerator"},
        "opt_strat": {
            "model": "GPClassificationModel",
            "generator": "OptimizeAcqfGenerator",
            "refit_every": 10,
        },
        "GPClassificationModel": {
            "inducing_size": 100,
            "mean_covar_factory": "default_mean_covar_factory",
            "inducing_point_method": "auto",
        },
        "default_mean_covar_factory": {
            "fixed_mean": False,
            "lengthscale_prior": "gamma",
            "outputscale_prior": "gamma",
            "kernel": "RBFKernel",
        },
        "OptimizeAcqfGenerator": {
            "acqf": [
                "LocalMI",
                "GlobalMI",
                "EAVC",
            ],
            "restarts": 2,
            "samps": 100,
        },
        # Add the probit transform for non-probit-specific acqfs
        "MCLevelSetEstimation": {"objective": "ProbitObjective"},
        "BernoulliMCMutualInformation": {"objective": "ProbitObjective"},
        "MCPosteriorVariance": {"objective": "ProbitObjective"},
    }

    for chunk in range(chunks):
        for problem in problems:
            out_fname = Path(f"{out_fname_base}/{problem.name}_chunk{chunk}_out.csv")
            intermediate_fname = Path(
                f"{out_fname_base}/{problem.name}_chunk{chunk}_checkpoint.csv"
            )
            print(f"starting {problem.name} benchmark... chunk {chunk} ")
            logger = BenchmarkLogger(log_every=log_frequency)
            benches = []
            for init in inits:
                local_config = deepcopy(bench_config)
                local_config["common"]["lb"] = str(problem.lb.tolist())
                local_config["common"]["ub"] = str(problem.ub.tolist())
                local_config["common"]["target"] = problem.threshold
                local_config["init_strat"]["n_trials"] = init
                local_config["opt_strat"]["n_trials"] = large_opt_size - init
                benches.append(
                    PathosBenchmark(
                        nproc=nproc,
                        problem=problem,
                        logger=logger,
                        configs=local_config,
                        global_seed=global_seed,
                        n_reps=reps_per_chunk,
                    )
                )
            bench = combine_benchmarks(*benches)
            bench.start_benchmarks()

            # checkpoint every minute in case something breaks
            while not bench.is_done:
                time.sleep(60)
                collate_start = time.time()
                print(
                    f"Checkpointing bench {problem} chunk {chunk}..., {len(bench.futures)}/{bench.num_benchmarks} alive"
                )
                bench.collate_benchmarks(wait=False)
                temp_results = bench.logger.pandas()
                if len(temp_results) > 0:
                    temp_results["rep"] = temp_results["rep"] + reps_per_chunk * chunk
                    temp_results["problem"] = problem.name
                    temp_results.to_csv(intermediate_fname)
                print(
                    f"Collate done in {time.time()-collate_start} seconds, {len(bench.futures)}/{bench.num_benchmarks} left"
                )

            print(f"Problem {problem} chunk {chunk} fully done!")
            final_results = bench.logger.pandas()
            final_results["rep"] = final_results["rep"] + reps_per_chunk * chunk
            final_results["problem"] = problem.name
            final_results.to_csv(out_fname)
