"""
Run a batch of experiments.

Creates

results/

    signflip/
    scaling/
    lie/
    ...

Each experiment stores:

    config.json
    round_metrics.csv
    client_scores.csv
    client_accuracy.csv
"""

import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import sys

sys.path.append(".")

from train import train, get_args
from experiments.config import CONFIG
from experiments.plotting import generate_all_plots

def save_history(history, args, save_dir):

    save_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Round metrics ----------------

    pd.DataFrame(
        history["round_metrics"]
    ).to_csv(

        save_dir / "round_metrics.csv",

        index=False
    )

    # ---------------- Client scores ----------------

    pd.DataFrame(
        history["client_scores"]
    ).to_csv(

        save_dir / "client_scores.csv",

        index=False
    )

    # ---------------- Client accuracy ----------------

    pd.DataFrame(
        history["client_accuracy"]
    ).to_csv(

        save_dir / "client_accuracy.csv",

        index=False
    )

    # ---------------- Config ----------------

    with open(save_dir / "config.json", "w") as f:

        json.dump(
            vars(args),
            f,
            indent=4
        )


def main():

    experiments = []

    for attack in CONFIG.attacks:

        for alpha in CONFIG.dirichlet_alpha:

            for byz in CONFIG.byzantine_clients:

                # Skip impossible setting
                if attack == "none" and byz != 0:
                    continue

                if attack != "none" and byz == 0:
                    continue

                for seed in CONFIG.seed:

                    experiments.append(
                        (
                            attack,
                            alpha,
                            byz,
                            seed
                        )
                    )

    print(f"\nTotal Experiments = {len(experiments)}\n")

    for attack, alpha, byz, seed in tqdm(experiments):

        # ------------------------------------
        # Reset argparse
        # ------------------------------------

        sys.argv = ["batchrun"]

        args = get_args()

        # ------------------------------------
        # Override parameters
        # ------------------------------------

        args.attack = attack

        args.dirichlet_alpha = alpha

        args.num_byzantine = byz

        args.seed = seed

        args.num_clients = 10

        args.CommunicationEpoch = CONFIG.communication_rounds

        # ------------------------------------
        # Run
        # ------------------------------------

        print("\n======================================")

        print(f"Attack      : {attack}")

        print(f"Alpha       : {alpha}")

        print(f"Byzantine   : {byz}")

        print(f"Seed        : {seed}")

        print("======================================")

        history = train(args)

        # ------------------------------------
        # Save
        # ------------------------------------

        save_dir = (

            Path("results")

            / attack

            / f"alpha_{alpha}"

            / f"byz_{byz}"

            / f"seed_{seed}"
        )

        save_history(

            history,

            args,

            save_dir
        )


        generate_all_plots(save_dir)

if __name__ == "__main__":

    main()