import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from sklearn.metrics import auc

from vpower.src.utils.assessment import f_beta_metrics, calc_uncertainty_regection_curve


def violin_plot(x: pd.Series, y: pd.Series, xlabel: str):
    """
    Creates a violin plot
    :param x: Input for x-axis
    :param y: Input for y-axis
    :param xlabel: Label to be used for the x-axis
    :return:
    """
    plt.figure(figsize=(7, 6))
    sns.violinplot(x=x, y=y, linewidth=1, scale="width", palette="pastel")
    plt.xlabel(xlabel)
    plt.ylabel("")
    plt.grid(axis="x")
    plt.show()
    plt.close()


def get_density_plots_matrix(data: pd.DataFrame, types_under_study: List[str], features_under_study: List[str],
                             colors_mapping: Dict, num_rows=4, num_cols=3):
    """
    Creates a matrix (num_rows, num_cols) of density plots of the features_under_study for the datasets defined by the
    type labels in types_under_study
    :param data: A pandas dataframe. It is required to have a column named 'type' that is a categorical variable
    indicating the type of individual datasets
    :param types_under_study: A list of the type labels to be plotted
    :param features_under_study: A list of features for which density plots are to be produced
    :param colors_mapping: A dictionary the maps each dataset type label at a given color
    :param num_rows: Number of rows of the matrix
    :param num_cols: Number of columns of the matrix
    :return:
    """
    plt.figure(figsize=(12, 14))
    plt.subplots_adjust(wspace=.4, hspace=.4)

    for i, f in enumerate(features_under_study):
        plt.subplot(num_rows, num_cols, i + 1)
        for t in types_under_study:
            mask = data["type"] == t
            data[mask][f].plot.kde(label=t, color=colors_mapping[t])
            plt.xlim((-1, 1) if f == "diff_speed_overground" else None)
            plt.title(f)
    plt.show()
    plt.close()


def plot_learning_curve(history: Dict, metric: str, label: str = None, ylims: tuple = None):
    """
    Creates the learning curves of the requested metric
    :param history: A dictionary of training histories of the individual ensemble members. History
     is the attribute of the history object produced during a keras model training that it is a record of training
     loss values and metrics values at successive epochs, as well as validation loss values and validation metrics
     values
    :param metric: Metric to be plotted
    :param label: X-axis label
    :param ylims: Y-axis limits
    :return:
    """
    plt.figure(figsize=(6, 4.5))
    for num, (k, hs) in enumerate(history.items()):
        plt.plot(np.arange(1, len(hs[metric]) + 1, 1) + 1,
                 hs[metric],
                 color=f"C{num}",
                 label=f"M {k} - train")
        plt.plot(np.arange(1, len(hs[metric]) + 1, 1) + 1,
                 hs[f"val_{metric}"],
                 color=f"C{num}",
                 linestyle="--",
                 label=f"M {k} - dev in")
    if len(history.keys()) > 1:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(loc="upper right")
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.grid()
    plt.ylim(ylims)
    plt.show()
    plt.close()


def get_comparison_error_retention_plot(error, uncertainty):
    """
    Creates an error retention plot for the three error ranking types: "random", "ordered" and "optimal"
    :param error: Per sample errors - array [n_samples]
    :param uncertainty: Uncertainties associated with each prediction. array [n_samples]
    :return:
    """
    for rank_type in ["random", "ordered", "optimal"]:
        rejection_mse = calc_uncertainty_regection_curve(errors=error,
                                                         uncertainty=uncertainty,
                                                         group_by_uncertainty=False,
                                                         rank_type=rank_type)
        retention_mse = rejection_mse[::-1]
        retention_fractions = np.linspace(0, 1, len(retention_mse))
        roc_auc = auc(x=retention_fractions[::-1], y=retention_mse)

        label = "Model" if rank_type == "ordered" else rank_type.capitalize()
        plt.plot(retention_fractions, retention_mse, label=f"{label} R-AUC:" + "{:.1e}".format(roc_auc))


def get_comparison_f1_retention_plot(error, uncertainty, threshold):
    """
    Creates a F1-score retention plot for the three error ranking types: "random", "ordered" and "optimal"
    :param error: Per sample errors - array [n_samples]
    :param uncertainty: Uncertainties associated with each prediction. array [n_samples]
    :param threshold: The error threshold below which we consider the prediction acceptable
    :return:
    """
    for rank_type in ["random", "ordered", "optimal"]:
        f_auc, f95, retention_f1 = f_beta_metrics(errors=error,
                                                  uncertainty=uncertainty,
                                                  threshold=threshold,
                                                  beta=1.0,
                                                  rank_type=rank_type)
        #         print(f"{rank_type}: F1 score at 95% retention: ", f95)
        retention_fractions = np.linspace(0, 1, len(retention_f1))

        label = "Model" if rank_type == "ordered" else rank_type.capitalize()
        plt.plot(retention_fractions, retention_f1, label=f"{label} F1-AUC:{np.round(f_auc, 3)}")
