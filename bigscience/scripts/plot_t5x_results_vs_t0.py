import csv
import json
import re
import subprocess
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--all', action="store_true", help="Plot all results in a single plot")
    parser.add_argument('--per-arch', action="store_true", help="Plot results grouped by architectures")
    parser.add_argument('--per-objective', action="store_true", help="Plots results grouped by objectives")
    parser.add_argument('--per-t0-adapted', action="store_true", help="Plots only T0 adapted models")
    args = parser.parse_args()

    assert args.all or args.per_arch or args.per_objective or args.per_t0_adapted

    return args

def load_t0_results(csv_path):
    with open(csv_path, "r") as f:
        return list(csv.DictReader(f))

def load_t5x_results(dir_path: Path):
    def remove_t0_eval(filename:str):
        name = filename.split("_t0_eval_")[0]
        name = name.replace("_bs2048", "")
        name = name.replace("_c4", "")
        return name

    all_results = {}
    for child in dir_path.iterdir():
        with open(child / "results.json", "r") as fi:
            results = json.load(fi)
        all_results[remove_t0_eval(child.name)] = results
    print(all_results.keys())
    return all_results

def get_experiment_name(filename: str):
    name = filename.replace("span_corruption", "SC")
    name = re.sub(r"^enc_dec", "ED", name)
    name = re.sub(r"^nc_dec", "NCD", name)
    name = re.sub(r"^c_dec", 'CD', name)
    name = name.replace("full_lm", "FLM")
    name = name.replace("prefix_lm", "PLM")
    name = re.sub(r"t0_adapt_([0-9]*)", r"T0(\1)", name)
    if name[:3] == "CD_":
        name = re.sub(r"lm_adapt_([0-9]*)", r"FLM(\1)", name)
    elif name[:4] == "NCD_" or name[:3] == "ED_":
        name = re.sub(r"lm_adapt_([0-9]*)", r"PLM(\1)", name)
    else:
        raise NotImplementedError
    name = name.replace("_", " + ")
    return name

TASKS = {
    'super_glue_copa': ('COPA', 0.5),
    'anli_r1': ('ANLI R1', 1/3),
    'anli_r2': ('ANLI R2', 1/3),
    'anli_r3': ('ANLI R3', 1/3),
    'super_glue_cb': ('CB', 1/3),
    'super_glue_rte': ('RTE', 0.5),
    'super_glue_wsc.fixed': ('WSC', 0.5),
    'winogrande_winogrande_xl': ('Winogrande', 0.5),
    'super_glue_wic': ('WiC', 0.5),
    'hellaswag': ('HellaSwag', 0.25),
    'story_cloze_2016': ('StoryCloze', 0.5),
}
def plot(t5x_data, t0_data):
    t5x_data, t5x_experiments = t5x_data
    assert len(TASKS) == 11
    fig, axs = plt.subplots(2, 6, figsize=(20, 8))
    axs = axs.flatten()

    task_median_score = {}
    for n, (task, (name, random_baseline)) in enumerate(TASKS.items()):
        t5lm_scores = [float(r["score"]) for r in t0_data
                       if r["runs"] == "xxl-lm-d4-091621"
                       and r["dataset_name"] == task
                       and r["metric_name"] == "accuracy (Rank)"
                       and r["score"]]
        t0_scores = [float(r["score"]) for r in t0_data
                     if r["runs"] == "xxl-lm-d4-091621-512"
                     and r["dataset_name"] == task
                     and r["metric_name"] == "accuracy (Rank)"
                     and r["score"]]
        t5x_scores = [[s["accuracy"] for k, s in t5x_data[name].items() if task.replace("anli_", "") in k]
                      for name in t5x_experiments]

        # Plot
        axs[n].axhline(100 * random_baseline, 0, len([t5lm_scores, t0_scores, *t5x_scores]))
        for i, scores in enumerate([t5lm_scores, t0_scores, *t5x_scores]):
            axs[n].scatter([i] * len(scores), scores, s=50, alpha=0.4)
        axs[n].set_title(name)

        # Gather median values
        task_median_score[task] = [random_baseline] + [np.median(scores) for scores in [t5lm_scores, t0_scores, *t5x_scores]]

    last_ax_id = len(TASKS) - 1
    axs[last_ax_id].legend(["T5+LM", "T0", *[get_experiment_name(exp) for exp in t5x_experiments]],
                           bbox_to_anchor=(1, 1), loc="upper left")
    for ax in axs[last_ax_id + 1:]:
        ax.set_visible(False)

    # ====== Plot agregated values =======
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    axs = axs.flatten()
    last_ax_id = 0

    # Plot average task median score
    average_task_media_score = np.mean(list(task_median_score.values()), axis=0)
    axs[last_ax_id].axhline(100 * average_task_media_score[0], 0, len(average_task_media_score) -1)
    for i, score in enumerate(average_task_media_score[1:]):
        axs[last_ax_id].scatter(i, score, s=50)
    axs[last_ax_id].set_title("Average of task median scores")

    axs[last_ax_id].legend(["Random", "T5+LM", "T0", *[get_experiment_name(exp) for exp in t5x_experiments]],
                           bbox_to_anchor=(1, 1), loc="upper left")
    for ax in axs[last_ax_id + 1:]:
        ax.set_visible(False)


def main():
    args = get_args()

    # Define directories
    results_dir = Path(__file__).resolve().parent.parent / "results" / "t0_eval"
    t0_results_dir = results_dir / "t0"
    t5x_results_dir = results_dir / "t5x"
    subprocess.run(["mkdir", "-p", t0_results_dir])
    subprocess.run(["mkdir", "-p", t5x_results_dir])

    # Sync previous results
    # gsutil cp gs://bigscience/experiment_d/aux_experiments/all_datasets_and_runs.csv ../results/t0_eval/t0
    if not (t0_results_dir / "all_datasets_and_runs.csv").exists():
        subprocess.run(["gsutil", "cp", "gs://bigscience/experiment_d/aux_experiments/all_datasets_and_runs.csv", t0_results_dir])
    # gsutil rsync -rd gs://bigscience-t5x/arch_objective_exps_v2/t0_eval ../results/t0_eval/t5x
    subprocess.run(["gsutil", "rsync", "-rd", "gs://bigscience-t5x/arch_objective_exps_v2/t0_eval", t5x_results_dir])

    # Load results
    t0_data = load_t0_results(t0_results_dir / "all_datasets_and_runs.csv")
    t5x_data = load_t5x_results(t5x_results_dir)

    # Plot results
    # We group experiments by:
    #  - objective
    #  - architecture
    LM_ADAPT_FROM = [28000, 30000]
    def key_architecture(experiment_name):
        if experiment_name[0] == 'c':
            return 0
        elif experiment_name[0] == 'n':
            return 1
        elif experiment_name[0] == 'e':
            return 2
        else:
            raise NotImplementedError
    def key_objective(experiment_name):
        suffixes = [
            "lm",
            *[f"lm_adapt_{lm_adapt}" for lm_adapt in LM_ADAPT_FROM],
            "lm_t0_adapt_32768",
            "span_corruption_t0_adapt_32768",
            *[f"lm_adapt_{lm_adapt}_t0_adapt_32768" for lm_adapt in LM_ADAPT_FROM]
        ]
        for i, suffix in enumerate(suffixes):
            if experiment_name.endswith(suffix):
                return i
        raise NotImplementedError(f"{experiment_name}")

    t5x_experiments = list(t5x_data.keys())
    # Define single ordering
    t5x_experiments = sorted(t5x_experiments, key=lambda x: (key_objective(x), key_architecture(x)))

    if args.all:
        plot((t5x_data, t5x_experiments), t0_data)

    def plot_per_group(group_fn):
        t5x_objective_keys = set(group_fn(x) for x in t5x_experiments)
        for group_id in t5x_objective_keys:
            t5x_experiments_per_group = [x for x in t5x_experiments if group_id == group_fn(x)]
            plot((t5x_data, t5x_experiments_per_group), t0_data)
    if args.per_objective:
        plot_per_group(key_objective)
    if args.per_arch:
        plot_per_group(key_architecture)
    if args.per_t0_adapted:
        def key_is_t0_adapted(experiment_name):
            return experiment_name.endswith("t0_adapt_32768")
        plot_per_group(key_is_t0_adapted)

    plt.show()
    print("Finished")

if __name__ == "__main__":
    main()