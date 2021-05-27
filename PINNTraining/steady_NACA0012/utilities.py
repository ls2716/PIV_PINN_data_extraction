import os

import matplotlib.pyplot as plt
import numpy as np


def set_directory(case_name):
    names = [f"{case_name}", f"{case_name}/plots", f"{case_name}/models"]
    for foldername in names:
        try:
            os.mkdir(foldername)
        except FileExistsError:
            pass


def plot_train_points(data, bc_ranges, bc_labels, case_name, title, figsize):
    plt.figure(figsize=figsize)
    # plt.title(f"Training points for {title}")
    plt.scatter(
        data.train_x[np.sum(data.num_bcs):, 0],
        data.train_x[np.sum(data.num_bcs):, 1],
        label="inside domain", s=0.15, marker='x'
    )
    bc_ranges = [0] + bc_ranges
    print('BC ranges', bc_ranges)
    print('Num bcs', data.num_bcs)
    for i in range(1, 2):
        plt.scatter(
            data.train_x[
                int(np.sum(data.num_bcs[: bc_ranges[i - 1]])): np.sum(
                    data.num_bcs[: bc_ranges[i]]
                ),
                0,
            ],
            data.train_x[
                int(np.sum(data.num_bcs[: bc_ranges[i - 1]])): np.sum(
                    data.num_bcs[: bc_ranges[i]]
                ),
                1,
            ],
            label=bc_labels[i - 1], s=1.5
        )
    for i in range(2, len(bc_ranges)):
        plt.scatter(
            data.train_x[
                int(np.sum(data.num_bcs[: bc_ranges[i - 1]])): np.sum(
                    data.num_bcs[: bc_ranges[i]]
                ),
                0,
            ],
            data.train_x[
                int(np.sum(data.num_bcs[: bc_ranges[i - 1]])): np.sum(
                    data.num_bcs[: bc_ranges[i]]
                ),
                1,
            ],
            label=bc_labels[i - 1], s=20
        )
    plt.legend(loc='right')
    plt.xlim(right=4.6)
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    axes=plt.gca()
    axes.set_aspect(1)
    plt.tight_layout()
    plt.savefig(f"{case_name}/training_points_{case_name}.png", dpi=400)
    plt.close()
