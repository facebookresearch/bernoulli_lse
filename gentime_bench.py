import os
import argparse


from copy import deepcopy
from pathlib import Path

global_seed = 1000
n_reps = 20

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Gentime Benchmarks")
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument(
        "--output_path", type=Path, default=Path("../data/gentime_bench")
    )
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.nproc)
    os.environ["MKL_NUM_THREADS"] = str(args.nproc)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.nproc)

    # multi-socket friendly args
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    os.environ["KMP_BLOCKTIME"] = str(args.nproc)
    import torch

    torch.set_num_interop_threads(args.nproc)
    torch.set_num_threads(args.nproc)

    from aepsych.benchmark import BenchmarkLogger, Benchmark

    from problems import (
        DiscrimLowDim,
        DiscrimHighDim,
        Hartmann6Binary,
    )

    out_fname_base = args.output_path
    out_fname_base.mkdir(
        parents=True, exist_ok=True
    )  # make an output folder if not exist
    problems = [
        DiscrimLowDim(),
        DiscrimHighDim(),
        Hartmann6Binary(),
    ]

    bench_config = {
        "common": {
            "outcome_type": "single_probit",
            "strategy_names": "[init_strat, opt_strat]",
        },
        "init_strat": {"n_trials": [10, 250, 500, 750], "generator": "SobolGenerator"},
        "opt_strat": {
            "n_trials": 2,
            "model": "GPClassificationModel",
            "generator": "OptimizeAcqfGenerator",
            "refit_every": 1,
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
    problem = problems[0]
    for problem in problems:
        out_fname = Path(f"{out_fname_base}/{problem.name}_{args.nproc}threads_out.csv")
        print(f"starting {problem.name} benchmark...")
        local_config = deepcopy(bench_config)
        local_config["common"]["lb"] = str(problem.lb.tolist())
        local_config["common"]["ub"] = str(problem.ub.tolist())
        local_config["common"]["target"] = problem.threshold
        logger = BenchmarkLogger(log_every=1)
        bench = Benchmark(
            problem=problem,
            logger=logger,
            configs=local_config,
            global_seed=global_seed,
            n_reps=n_reps,
        )

        bench.run_benchmarks()

        print(f"Problem {problem} fully done!")
        final_results = bench.logger.pandas()
        final_results["problem"] = problem.name
        final_results["nproc"] = args.nproc
        final_results.to_csv(out_fname)
