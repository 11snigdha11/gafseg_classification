"""
experiments/logger.py

Utility functions for saving experiment results.
"""

import json
from pathlib import Path

import pandas as pd


def create_experiment_dir(base_dir, args):
    """
    Creates directory:

    results/
        attack/
            alpha_x/
                byz_x/
                    seed_x/
    """

    exp_dir = (
        Path(base_dir)
        / args.attack
        / f"alpha_{args.dirichlet_alpha}"
        / f"byz_{args.num_byzantine}"
        / f"seed_{args.seed}"
    )

    exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir


def save_history(history, exp_dir):
    """
    Save experiment history to CSV files.
    """

    pd.DataFrame(
        history["round_metrics"]
    ).to_csv(
        exp_dir / "round_metrics.csv",
        index=False
    )

    pd.DataFrame(
        history["client_scores"]
    ).to_csv(
        exp_dir / "client_scores.csv",
        index=False
    )

    pd.DataFrame(
        history["client_accuracy"]
    ).to_csv(
        exp_dir / "client_accuracy.csv",
        index=False
    )


def save_config(args, exp_dir):
    """
    Save experiment configuration.
    """

    with open(exp_dir / "config.json", "w") as f:

        json.dump(
            vars(args),
            f,
            indent=4
        )


def save_summary(history, exp_dir):
    """
    Save one-line summary for this experiment.
    """

    summary = {}

    summary["attack"] = history["round_metrics"][0]["attack"]

    summary["seed"] = history["round_metrics"][0]["seed"]

    summary["alpha"] = history["round_metrics"][0]["alpha"]

    summary["num_byzantine"] = history["round_metrics"][0]["num_byzantine"]

    acc = [
        x["test_accuracy"]
        for x in history["round_metrics"]
    ]

    summary["best_test_accuracy"] = max(acc)

    summary["final_test_accuracy"] = acc[-1]

    pd.DataFrame([summary]).to_csv(
        exp_dir / "summary.csv",
        index=False
    )