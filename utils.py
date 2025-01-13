"""
Utility functions for cross-validation and plotting.

These routines were written by Rick van Veen in 
connection to van Veen, Rick, et al. 
"Subspace corrected relevance learning with application in neuroimaging."
Artificial Intelligence in Medicine 149 (2024): 102786.
"""

import numpy as np
import matplotlib.pyplot as plt
import settings

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
)

def transform(
    data: np.ndarray,
    omega_hat: np.ndarray,
    eigenvectors: np.ndarray,
    scale: bool = True,
):
    if scale:
        return np.matmul(data, omega_hat.T)

    return np.matmul(data, eigenvectors.T)


def cross_validation(pipeline, cv, data, labels):
    """
    Performs cross validation and returns the averaged model parameters.

    It is assumed that the model is GMLVQ and is accessible by "gmlvq" in the pipeline.
    """
    n_iterations = cv.get_n_splits()

    cv_lambda = np.zeros((data.shape[1], data.shape[1], n_iterations))
    cv_auc = np.zeros(n_iterations)
    cv_auc_train = np.zeros(n_iterations)

    cv_accuracy = np.zeros(n_iterations)
    cv_balanced_accuracy = np.zeros(n_iterations)
    cv_balanced_accuracy_train = np.zeros(n_iterations)

    classes = np.unique(labels)
    cv_confmat = np.zeros((classes.size, classes.size, n_iterations))

    multi_class = "raise"
    if np.size(classes) > 2:
        multi_class = "ovr"

    for i, (train_index, test_index) in enumerate(cv.split(data, labels)):
        X_train, X_test = data[train_index, :], data[test_index, :]
        y_train, y_test = labels[train_index], labels[test_index]

        pipeline = pipeline.fit(X_train, y_train)
        
        if np.size(classes) > 2:
            y_test_score = pipeline.predict_proba(X_test)
            y_train_score = pipeline.predict_proba(X_train)
        else:
            y_test_score = pipeline.decision_function(X_test)
            y_train_score = pipeline.decision_function(X_train)

        # Training scores
        y_train_pred = pipeline.predict(X_train)
        cv_auc_train[i] = roc_auc_score(y_train, y_train_score, multi_class=multi_class)

        cv_balanced_accuracy_train[i] = balanced_accuracy_score(y_train, y_train_pred)

        # Test scores
        y_test_pred = pipeline.predict(X_test)
        cv_auc[i] = roc_auc_score(y_test, y_test_score, multi_class=multi_class)

        cv_confmat[:, :, i] = confusion_matrix(y_test, y_test_pred, labels=classes, normalize = 'true')
        cv_balanced_accuracy[i] = balanced_accuracy_score(y_test, y_test_pred)
        cv_accuracy[i] = accuracy_score(y_test, y_test_pred)
        
        cv_lambda[:, :, i] = pipeline.named_steps["gmlvq"].lambda_

    return (
        cv_lambda,
        cv_auc,
        cv_confmat,
        cv_balanced_accuracy,
        cv_accuracy,
        cv_auc_train,
        cv_balanced_accuracy_train,
    )


def plot_projection(
    projected_data,
    projected_labels,
    projected_model,
    projected_model_labels,
    add_legend=True,
    axis=None,
):
    fig, ax = plt.subplots(figsize = (5, 3.5))
    ax.margins(x=0.1, y=0.1)

    _add_data_projection(
        projected_data, projected_labels, ax, settings.MARKER_STYLES, axis=axis
    )
    _add_data_projection(
        projected_model,
        projected_model_labels,
        ax,
        settings.PROTOTYPE_MARKER_STYLE,
        axis=axis,
    )

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    _add_decision_voronoi(
        projected_model,
        projected_model_labels,
        ax,
        settings.COLORDICT,
        axis=axis,
    )

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    if add_legend:
        _add_legend(np.unique(projected_labels), ax, settings.LEGEND_MARKERS, ncols=3)

    ax.set_xlabel(rf"Proj. on 1st eigenvector")
    ax.set_ylabel(rf"Proj. on 2nd eigenvector")

    ax.grid(True)
    ax.set_axisbelow(True)

    return fig, ax


def _add_data_projection(
    data,
    labels,
    ax,
    styles,
    axis=None,
):
    if axis is None:
        axis = (0, 1)

    unique_labels = np.unique(labels)
    for label in unique_labels:
        ii = label == np.asarray(labels)
        ax.scatter(
            data[ii, axis[0]],
            data[ii, axis[1]],
            **styles[label],
            label=label,
        )
        
        
def _add_legend(legend_labels, ax, legend_styles, ncols=None):
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            **legend_styles[label],
            label=f"{label}",
        )
        for label in legend_labels
    ]
    if ncols is None:
        ncols = np.size(legend_labels)

    legend = plt.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=ncols,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        borderaxespad=0,
        fontsize = 14,
    )
    return legend


def _add_decision_voronoi(projected_prototypes, projected_prototypes_labels, ax, colors, axis=None):
    from scipy.spatial import Voronoi, voronoi_plot_2d

    if axis is None:
        axis = (0,1)

    voronoi_points = np.append(
        projected_prototypes[:, axis],
        [[999, 999], [-999, 999], [999, -999], [-999, -999]],
        axis=0,
    )
    voronoi = Voronoi(voronoi_points)

    voronoi_plot_2d(voronoi, ax=ax, show_vertices=False, show_points=False)

    for point_index, region_index in enumerate(voronoi.point_region[:-4]):
        region = voronoi.regions[region_index]
        if not -1 in region:
            color = colors[projected_prototypes_labels[point_index]]
            polygon = [voronoi.vertices[i] for i in region]
            ax.fill(*zip(*polygon), color=color, zorder=-10, alpha=0.2)
        