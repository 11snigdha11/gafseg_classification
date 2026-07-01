"""
experiments/plotting.py

Generate plots from saved experiment CSV files.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------
# Common Plot Style
# ---------------------------------------------------------

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.size"] = 12


# ---------------------------------------------------------
# Accuracy
# ---------------------------------------------------------

def plot_accuracy(round_df, save_dir):

    plt.figure()

    plt.plot(
        round_df["round"],
        round_df["test_accuracy"],
        marker="o"
    )

    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Global Test Accuracy")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        Path(save_dir) / "accuracy.png",
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# Raw Scores
# ---------------------------------------------------------

def plot_raw_scores(score_df, save_dir):

    plt.figure()

    for client in sorted(score_df["client"].unique()):

        df = score_df[
            score_df["client"] == client
        ]

        plt.plot(
            df["round"],
            df["raw_score"],
            label=f"Client {client}"
        )

    plt.xlabel("Communication Round")
    plt.ylabel("Raw Score")
    plt.title("Client Raw Scores")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        Path(save_dir) / "raw_scores.png",
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# Softmax Weights
# ---------------------------------------------------------

def plot_softmax(score_df, save_dir):

    plt.figure()

    for client in sorted(score_df["client"].unique()):

        df = score_df[
            score_df["client"] == client
        ]

        plt.plot(
            df["round"],
            df["softmax_weight"],
            label=f"Client {client}"
        )

    plt.xlabel("Communication Round")
    plt.ylabel("Softmax Weight")
    plt.title("Aggregation Weights")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        Path(save_dir) / "softmax_weights.png",
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# Update Norms
# ---------------------------------------------------------

def plot_update_norm(score_df, save_dir):

    plt.figure()

    for client in sorted(score_df["client"].unique()):

        df = score_df[
            score_df["client"] == client
        ]

        plt.plot(
            df["round"],
            df["update_norm"],
            label=f"Client {client}"
        )

    plt.xlabel("Communication Round")
    plt.ylabel("Update Norm")
    plt.title("Client Update Norms")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        Path(save_dir) / "update_norms.png",
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# Client Train Accuracy
# ---------------------------------------------------------

def plot_client_accuracy(acc_df, save_dir):

    plt.figure()

    for client in sorted(acc_df["client"].unique()):

        df = acc_df[
            acc_df["client"] == client
        ]

        plt.plot(
            df["round"],
            df["train_accuracy"],
            label=f"Client {client}"
        )

    plt.xlabel("Communication Round")
    plt.ylabel("Train Accuracy (%)")
    plt.title("Client Training Accuracy")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        Path(save_dir) / "client_accuracy.png",
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# Generate everything
# ---------------------------------------------------------

def generate_all_plots(save_dir):

    save_dir = Path(save_dir)

    round_df = pd.read_csv(
        save_dir / "round_metrics.csv"
    )

    score_df = pd.read_csv(
        save_dir / "client_scores.csv"
    )

    acc_df = pd.read_csv(
        save_dir / "client_accuracy.csv"
    )

    plot_accuracy(
        round_df,
        save_dir
    )

    plot_raw_scores(
        score_df,
        save_dir
    )

    plot_softmax(
        score_df,
        save_dir
    )

    plot_update_norm(
        score_df,
        save_dir
    )

    plot_client_accuracy(
        acc_df,
        save_dir
    )

    print(f"Plots saved to {save_dir}")