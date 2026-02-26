import matplotlib.pyplot as plt
import numpy as np


def main():
    methods = [
        "Noisy Image",
        "ADMM-TV",
        "Direct CNN",
        "ADMM-PnP-DL (U-Net)",
        "ADMM-PnP-DL (DnCNN)",
    ]

    # Midpoints of the ranges from the table in the report
    psnr = [17.5, 25.0, 27.5, 31.5, 30.5]      # dB
    ssim = [0.40, 0.68, 0.78, 0.88, 0.84]      # unitless
    enl = [1.5, 4.0, 6.0, 10.0, 8.0]           # unitless
    runtime = [0.0, 3.5, 0.3, 1.25, 0.9]       # seconds

    x = np.arange(len(methods))
    width = 0.2

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # PSNR, SSIM, ENL on left y-axis
    ax1.bar(x - width, psnr, width, label="PSNR (dB)")
    ax1.bar(x, ssim, width, label="SSIM")
    ax1.bar(x + width, enl, width, label="ENL")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=30, ha="right")
    ax1.set_ylabel("PSNR / SSIM / ENL")
    ax1.set_title("Summary Performance Comparison")
    ax1.grid(axis="y", alpha=0.3)

    # Runtime on second axis
    ax2 = ax1.twinx()
    ax2.plot(x, runtime, "r--o", label="Runtime (s)")
    ax2.set_ylabel("Runtime (s)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig("summary_performance_bar_chart.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()

{
  "cells": [],
  "metadata": {
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 2
}