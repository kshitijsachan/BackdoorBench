import argparse
import os
from typing import Optional

import torch
from matplotlib import pyplot as plt


def get_threshold(clean_scores: list[float], bd_scores: list[float], fpr: Optional[float], fnr: Optional[float]):
    if isinstance(fpr, float):
        assert 0 < fpr < 1, "fpr must be between 0 and 1."
        clean_sorted = sorted(clean_scores)
        # round down using int instead of round to nearest integer because don't want index out of bound error
        return clean_sorted[int((1 - fpr) * len(clean_sorted))]
    elif isinstance(fnr, float):
        assert 0 < fnr < 1, "fnr must be between 0 and 1."
        bd_sorted = sorted(bd_scores)
        return bd_sorted[int(fnr * len(bd_sorted))]
    else:
        raise ValueError("fpr and fnr cannot be both None.")


def compute_asr(bd_scores: list[float], threshold: float):
    bd_incorrect = sum(score < threshold for score in bd_scores)
    return bd_incorrect / len(bd_scores)


def compute_clean_accuracy(clean_scores: list[float], clean_accuracy: list[bool], threshold: float):
    clean_correct = sum((score < threshold) and acc for (score, acc) in zip(clean_scores, clean_accuracy))
    return clean_correct / len(clean_scores)


def plot_scores(scores: dict[str, list[float]], vlines: Optional[dict[str, float]], title: str = ""):
    fig, ax = plt.subplots()
    for name, score_list in scores.items():
        ax.hist(score_list, label=name, alpha=0.5, bins=30)
    if vlines is not None:
        for name, x_val in vlines.items():
            ax.axvline(x_val, color="k", linestyle="dashed", linewidth=1, label=name)
    if title:
        ax.set_title(title)
    ax.set_xlabel("anomaly score")
    ax.set_ybound(0, 1000)
    # ax.set_yscale("log")
    ax.set_xbound(-0.1, 14)
    fig.legend()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, required=True, help="name of the experiment folder")
    parser.add_argument("--experiment_name", type=str, help="False Negative Rate")
    parser.add_argument("--defense", type=str, required=True, help="name of the defense")
    parser.add_argument("--rrfs", action="store_true", help="load data and save files to rrfs instead of locally")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fpr", type=float, help="False Positive Rate")
    group.add_argument("--fnr", type=float, help="False Negative Rate")
    args = parser.parse_args()

    bdbench_root_dir = os.path.expanduser("~/rrfs/ksachan/backdoor_bench") if args.rrfs else "."
    results_dir = os.path.join(
        bdbench_root_dir, "record", args.result_file, "defense", args.defense, args.experiment_name
    )
    result = torch.load(f"{results_dir}/result.pt")
    clean_scores, bd_scores, clean_is_correct = result["clean_scores"], result["bd_scores"], result["clean_is_correct"]
    assert isinstance(clean_is_correct[0], bool)
    print("TODO: remove this assert")
    threshold = get_threshold(clean_scores, bd_scores, args.fpr, args.fnr)
    asr = compute_asr(bd_scores, threshold)
    clean_accuracy = compute_clean_accuracy(clean_scores, clean_is_correct, threshold)
    print("Num clean examples", len(clean_scores))
    print("Num bd examples", len(bd_scores))
    print(f"threshold: {threshold:.4f}")
    print(f"asr: {asr * 100:.2f}%")
    print(f"clean_accuracy: {clean_accuracy * 100:.2f}%")
    print(f"Clean score avg: {sum(clean_scores) / len(clean_scores):.3f}")
    print(f"Bd score avg: {sum(bd_scores) / len(bd_scores):.3f}")
    print(f"Clean score median: {sorted(clean_scores)[len(clean_scores) // 2]:.3f}")
    print(f"Bd score median: {sorted(bd_scores)[len(bd_scores) // 2]:.3f}")
    torch.save(
        result | {"threshold": threshold, "asr": asr, "clean_accuracy": clean_accuracy},
        f"{results_dir}/result.pt",
    )
    vline_label = f"fpr = {args.fpr * 100}%" if args.fpr is not None else f"fnr = {args.fnr * 100}%"
    fig = plot_scores(
        {"clean": clean_scores, "bd": bd_scores}, {vline_label: threshold}, "Fine Tuning + Reg anomaly scores"
    )
    fig.savefig(f"{results_dir}/anomaly_score.png")
    print("Plotted kl divergence scores.")
