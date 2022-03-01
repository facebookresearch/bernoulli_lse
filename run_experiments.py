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

from aepsych.benchmark import BenchmarkLogger, PathosBenchmark

from problems import (
    DiscrimLowDim,
    DiscrimHighDim,
    Hartmann6Binary,
    ContrastSensitivity6d,  # This takes a few minutes to instantiate due to fitting the model
)

problem_map = {
    "discrim_lowdim": DiscrimLowDim,
    "discrim_highdim": DiscrimHighDim,
    "hartmann6_binary": Hartmann6Binary,
    "contrast_sensitivity_6d": ContrastSensitivity6d,
}


def make_argparser():
    parser = argparse.ArgumentParser(description="Lookahead LSE Benchmarks")
    parser.add_argument("--nproc", type=int, default=124)
    parser.add_argument("--reps_per_chunk", type=int, default=20)
    parser.add_argument("--chunks", type=int, default=15)
    parser.add_argument("--large_opt_size", type=int, default=740)
    parser.add_argument("--small_opt_size", type=int, default=490)
    parser.add_argument("--init_size", type=int, default=10)
    parser.add_argument("--global_seed", type=int, default=1000)
    parser.add_argument("--log_frequency", type=int, default=10)
    parser.add_argument("--output_path", type=Path, default=Path("../data/cameraready"))
    parser.add_argument(
        "--problem",
        type=str,
        choices=[
            "discrim_highdim",
            "discrim_lowdim",
            "hartmann6_binary",
            "contrast_sensitivity_6d",
            "all",
        ],
        default="all",
    )
    return parser


if __name__ == "__main__":

    parser = make_argparser()
    args = parser.parse_args()
    out_fname_base = args.output_path
    out_fname_base.mkdir(
        parents=True, exist_ok=True
    )  # make an output folder if not exist
    if args.problem == "all":
        problems = [
            DiscrimLowDim(),
            DiscrimHighDim(),
            Hartmann6Binary(),
            ContrastSensitivity6d(),
        ]
    else:
        problems = [problem_map[args.problem]()]

    bench_config = {
        "common": {
            "outcome_type": "single_probit",
            "strategy_names": "[init_strat, opt_strat]",
        },
        "init_strat": {"n_trials": args.init_size, "generator": "SobolGenerator"},
        "opt_strat": {
            "model": "GPClassificationModel",
            "generator": "OptimizeAcqfGenerator",
            "refit_every": args.log_frequency,
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
                "MCLevelSetEstimation",  # Straddle
                "LocalSUR",
                "GlobalMI",
                "GlobalSUR",
                "EAVC",
                "ApproxGlobalSUR",
                "MCPosteriorVariance",  # BALV
                "BernoulliMCMutualInformation",  # BALD
            ],
            "restarts": 2,
            "samps": 100,
        },
        # Add the probit transform for non-probit-specific acqfs
        "MCLevelSetEstimation": {"objective": "ProbitObjective"},
        "BernoulliMCMutualInformation": {"objective": "ProbitObjective"},
        "MCPosteriorVariance": {"objective": "ProbitObjective"},
    }

    for chunk in range(args.chunks):
        for problem in problems:
            out_fname = Path(f"{out_fname_base}/{problem.name}_chunk{chunk}_out.csv")
            intermediate_fname = Path(
                f"{out_fname_base}/{problem.name}_chunk{chunk}_checkpoint.csv"
            )
            print(f"starting {problem.name} benchmark... chunk {chunk} ")
            local_config = deepcopy(bench_config)
            local_config["common"]["lb"] = str(problem.lb.tolist())
            local_config["common"]["ub"] = str(problem.ub.tolist())
            local_config["common"]["target"] = problem.threshold
            local_config["opt_strat"]["n_trials"] = (
                args.small_opt_size
                if problem.name == "discrim_lowdim"
                else args.large_opt_size
            )
            logger = BenchmarkLogger(log_every=args.log_frequency)
            acq_bench = PathosBenchmark(
                nproc=args.nproc,
                problem=problem,
                logger=logger,
                configs=local_config,
                global_seed=args.global_seed,
                n_reps=args.reps_per_chunk,
            )
            sobol_config = deepcopy(local_config)
            sobol_config["opt_strat"]["generator"] = "SobolGenerator"
            del sobol_config["OptimizeAcqfGenerator"]
            sobol_bench = PathosBenchmark(
                nproc=args.nproc,
                problem=problem,
                logger=logger,
                configs=sobol_config,
                global_seed=args.global_seed,
                n_reps=args.reps_per_chunk,
            )
            bench = acq_bench + sobol_bench
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
                    temp_results["rep"] = (
                        temp_results["rep"] + args.reps_per_chunk * chunk
                    )
                    temp_results["problem"] = problem.name
                    temp_results.to_csv(intermediate_fname)
                print(
                    f"Collate done in {time.time()-collate_start} seconds, {len(bench.futures)}/{bench.num_benchmarks} left"
                )

            print(f"Problem {problem} chunk {chunk} fully done!")
            final_results = bench.logger.pandas()
            final_results["rep"] = final_results["rep"] + args.reps_per_chunk * chunk
            final_results["problem"] = problem.name
            final_results.to_csv(out_fname)
