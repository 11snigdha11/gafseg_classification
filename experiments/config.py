"""
Experiment configuration.

This file ONLY specifies which experiments to run.

All training hyperparameters remain inside get_args().
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:

    # ======================================================
    # Experiment Grid
    # ======================================================

    # attacks = [

    #     "none",

    #     "signflip",

    #     "scaling",

    #     "gaussian",

    #     "random",

    #     "lie",

    #     "minmax",

    #     "minsum",
    # ]


    # byzantine_clients = [

    #     1,

    #     2,

    #     3,
    # ]


    # dirichlet_alpha = [
    #     10,

    #     1.0,

    #     0.5,

    #     0.1,
    # ]


    # seed = [

    #     1,

    #     2,

    
    # ]
    attacks=['scaling']
    byzantine_clients=[1]
    dirichlet_alpha=[0.5]
    seed=[1]


    communication_rounds = 100
    # # Later change to 100.
    # attacks = ["signflip","lie"]
    # byzantine_clients = [1]
    # dirichlet_alpha = [0.5]
    # random_seeds = [1]
    # communication_rounds = 5

    # ======================================================
    # Logging
    # ======================================================

    save_every_round = True

    statistics_every = 10

    checkpoint_every = 10

    save_plots = True

    save_csv = True

    save_json = True

    save_logs = True


    # ======================================================
    # Output
    # ======================================================

    results_dir = Path("results")

    dpi = 300


CONFIG = ExperimentConfig()