import numpy as np 
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.patches

import colorama
import matplotlib.pyplot as plt

import pandas as pd

from pf_utils import ParticleCloud
import pf_utils
import settings 
import data_generation_utils


# Support for a standard set of nice colours. Some colours 
# are obtained from the XKCD colours list: https://xkcd.com/color/rgb/

BLACK   = "#000000"
BLUE    = "#030aa7" # XKCD: "Cobalt blue"
RED     = "#9a0200" # XKCD: "Deep red"
GREEN   = "#02590f" # XKCD: "Deep green"
GREY    = "#929591" # XKCD: "Grey"

STANDARD_COLOURS = [BLUE, RED, GREEN]
def get_standard_colour(i):
    if i < len(STANDARD_COLOURS):
        return STANDARD_COLOURS[i]
    return BLACK

# Standard line plotting
PRIMARY_LINESTYLE = "-" 
PRIMARY_LINEWIDTH = 2

SECONDARY_LINESTYLE = "-"
SECONDARY_LINEWIDTH = 1

FINE_LINE = 0.25
FINE_SCATTER_SIZE = 4


# Error bar plotting
GAUSSIAN_CONFIDENCE_95 = 1.96 # The number of standard deviations within which 95% of all data must lie for a Gaussian

ERROR_BAR_LINESTYLE = "--"
ERROR_BAR_LINEWIDTH = 1

# Reference value plotting
REFERENCE_VALUE_LINESTYLE = "--"
REFERENCE_VALUE_LINEWIDTH = 1

# Clutter plotting
CLUTTER_MARKER = "2"

# Standard styles for kwargs
PRIMARY_STYLE_KWARGS = {
    "linestyle": PRIMARY_LINESTYLE,
    "linewidth": PRIMARY_LINEWIDTH
}
PRIMARY_SCATTER_STYLE_KWARGS = {
    "s": 25,
    "marker": "."
}

SECONDARY_STYLE_KWARGS = {
    "linestyle": SECONDARY_LINESTYLE,
    "linewidth": SECONDARY_LINEWIDTH
}
SECONDARY_SCATTER_STYLE_KWARGS = {
    "s": 9,
    "marker": "."
}
CLUTTER_SCATTER_STYLE_KWARGS = {
    "s": 25,
    "marker": CLUTTER_MARKER,
    "c": GREY,
    "alpha": 1,
}

ERROR_BAR_STYLE_KWARGS = {
    "linestyle": ERROR_BAR_LINESTYLE,
    "linewidth": ERROR_BAR_LINEWIDTH,
    "color": GREY
}

REFERENCE_VALUE_STYLE_KWARGS = {
    "linestyle": REFERENCE_VALUE_LINESTYLE,
    "linewidth": REFERENCE_VALUE_LINEWIDTH,
}

BIG_SCATTER_STYLE_KWARGS = {
    "s": 400,
    "marker": ".",
}

# Display size details
MM_PER_INCH = 25.6

A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297
A4_WIDTH_INCH = A4_WIDTH_MM / MM_PER_INCH
A4_HEIGHT_INCH = A4_HEIGHT_MM / MM_PER_INCH

STANDARD_TEXT_WIDTH_INCH = A4_WIDTH_INCH - 2

BIG_FIGURE_WIDTH = 13


# Preparing axes
def prepare_axis(axs : plt.axis, title : str, x_label : str, y_label : str, legend_on : bool):
    new_axs : plt.axis = axs 
    if title is not None and len(title) > 0:
        new_axs.set_title(title)
    if x_label is not None and len(x_label) > 0:
        new_axs.set_xlabel(x_label)
    if y_label is not None and len(y_label) > 0:
        new_axs.set_ylabel(y_label)

    if legend_on:
        new_axs.legend()
    
    return new_axs

# Preparing figures
def prepare_figure(fig : plt.figure, title : str, width, height, parameters):
    if title is not None:
        fig.suptitle(title)
    if width is not None:
        fig.set_figwidth(width)
    if height is not None:
        fig.set_figheight(height)
    if parameters is not None:
        add_parameter_signature(fig, parameters)
    fig.tight_layout()

# Saving figures
FILE_TYPES = ["pdf", "png"]
STANDARD_FILE_TYPE = FILE_TYPES[0]
def save_figure(fig : plt.figure, directory_path : str, file_name : str | None, file_type : str = STANDARD_FILE_TYPE):
    if file_name is None:
        print(f"{colorama.Fore.LIGHTYELLOW_EX}File name is None, skipping saving.")
    else:
        fig.savefig(directory_path + "/" + file_name + "." + file_type, bbox_inches='tight')




def plot_trajectory_plot(zn_mean_history, data : data_generation_utils.MeasurementSet, ground_truth_position : np.ndarray, params : settings.TrackerParameters, file_name : str, burn_in=0):
    uncluttered_data = data.get_uncluttered_data()
    clutter_history = data.get_clutter_history()
    true_measurement_observed = np.array([dat.size > 0 for dat in uncluttered_data])
    
    # Plot trajectory plot
    fig, axs = plt.subplots(nrows=1,ncols=1)
    axs.plot(ground_truth_position[:,0], ground_truth_position[:,1], label="Ground truth", **SECONDARY_STYLE_KWARGS, c=get_standard_colour(1))
    
    for ts in range(params.simulation_time):
        if true_measurement_observed[ts]:
            axs.scatter(uncluttered_data[ts][0,0], uncluttered_data[ts][0,1], label="Measurements" if ts == 0 else None, **SECONDARY_SCATTER_STYLE_KWARGS, c=BLACK)
    for ts in range(params.simulation_time):
        if clutter_history[ts].size == 1:
            continue
        axs.scatter(clutter_history[ts][:,0], clutter_history[ts][:,1].flatten(), label="Clutter" if ts==0 else None, **CLUTTER_SCATTER_STYLE_KWARGS)
    if zn_mean_history is not None:
        axs.plot(zn_mean_history[burn_in:,0], zn_mean_history[burn_in:,1], label="Estimate", **PRIMARY_STYLE_KWARGS, c=get_standard_colour(0))
    axs.set_aspect("equal")
    axs = prepare_axis(axs, None, "$x$", "$y$", True)
    prepare_figure(fig, None, None, None, None)
    if file_name is not None:
        save_figure(fig, "output", file_name)
    fig.show()


def plot_ar_coefficient_estimates(a_mean_history : np.ndarray, a_cov_history : np.ndarray, true_process_coefficients : np.ndarray | None, file_name : str):
    model_order = a_mean_history.shape[1] if a_mean_history is not None else true_process_coefficients.shape[1]
    fig, axs = plt.subplots(nrows=model_order, ncols=1, sharex="all", squeeze=False)
    for i in range(model_order):
        this_ax = axs[i,0]
        if true_process_coefficients is not None:
            this_ax.plot(true_process_coefficients[model_order:, i], c=get_standard_colour(1), label="Ground truth", **PRIMARY_STYLE_KWARGS)

        if a_mean_history is not None:
            this_ax.plot(a_mean_history[model_order:, i], c=BLACK, label="Mean", **PRIMARY_STYLE_KWARGS)
            this_ax.plot(a_mean_history[model_order:, i] + GAUSSIAN_CONFIDENCE_95 * np.sqrt(a_cov_history[model_order:, i, i]), label="95% confidence", **ERROR_BAR_STYLE_KWARGS)
            this_ax.plot(a_mean_history[model_order:, i] - GAUSSIAN_CONFIDENCE_95 * np.sqrt(a_cov_history[model_order:, i, i]), **ERROR_BAR_STYLE_KWARGS)
        this_ax = prepare_axis(this_ax, "", "Time" if i == model_order - 1 else "", "$a_{" + str(i + 1) + "}$", i == 0)
    prepare_figure(fig, None, None, None, None)
    save_figure(fig, "output", file_name)
    fig.show()


def plot_ar_coefficient_estimates_with_cp(a_mean_history : np.ndarray, a_cov_history : np.ndarray, true_process_coefficients : np.ndarray | None, file_name : str):
    model_order = a_mean_history.shape[1] if a_mean_history is not None else true_process_coefficients.shape[1]
    fig, axs = plt.subplots(nrows=model_order, ncols=1, sharex="all", squeeze=False)
    for i in range(model_order):
        this_ax = axs[i,0]
        if true_process_coefficients is not None:
            this_ax.plot(true_process_coefficients[model_order:, i], c=get_standard_colour(1), label="WNA characteristic polynomial", **PRIMARY_STYLE_KWARGS)

        if a_mean_history is not None:
            this_ax.plot(a_mean_history[model_order:, i], c=BLACK, label="Mean", **PRIMARY_STYLE_KWARGS)
            this_ax.plot(a_mean_history[model_order:, i] + GAUSSIAN_CONFIDENCE_95 * np.sqrt(a_cov_history[model_order:, i, i]), label="95% confidence", **ERROR_BAR_STYLE_KWARGS)
            this_ax.plot(a_mean_history[model_order:, i] - GAUSSIAN_CONFIDENCE_95 * np.sqrt(a_cov_history[model_order:, i, i]), **ERROR_BAR_STYLE_KWARGS)
        this_ax = prepare_axis(this_ax, "", "Time" if i == model_order - 1 else "", "$a_{" + str(i + 1) + "}$", i == 0)
    prepare_figure(fig, None, None, None, None)
    save_figure(fig, "output", file_name)
    fig.show()


def plot_pole_trajectory(true_process_coefficients, file_name : str):
    fig, axs = plt.subplots()

    def get_roots(coefs):
        return np.roots(np.concatenate([np.ones(1), -coefs]))

    process_poles = np.zeros(true_process_coefficients.shape, dtype="complex")
    for ts in range(true_process_coefficients.shape[0]):
        process_poles[ts, :] = get_roots(true_process_coefficients[ts, :])

    unit_circle = matplotlib.patches.Ellipse(xy=(0, 0), width=2, height=2)
    unit_circle.set_facecolor("#fff")
    unit_circle.set_edgecolor(BLACK)
    unit_circle.set_linewidth(SECONDARY_LINEWIDTH * 0.5)
    unit_circle.set_linestyle(SECONDARY_LINESTYLE)
    unit_circle.set_alpha(1)
    unit_circle.set_fill(False)

    axs.axhline(0, linestyle=SECONDARY_LINESTYLE, linewidth=SECONDARY_LINEWIDTH * 0.5, c=BLACK)
    axs.axvline(0, linestyle=SECONDARY_LINESTYLE, linewidth=SECONDARY_LINEWIDTH * 0.5, c=BLACK)

    axs.add_artist(unit_circle)
    axs.set_xlim([-1.1, 1.1])
    axs.set_xticks([-1, -0.5, 0, 0.5, 1])
    axs.set_ylim([-1.1, 1.1])
    axs.set_yticks([-1, -0.5, 0, 0.5, 1])
    axs.set_aspect("equal")

    for i in range(true_process_coefficients.shape[1]):
        these_poles = process_poles[:, i].flatten()
        colour_choices = np.where(np.abs(np.imag(these_poles)) < 1E-3, 0, np.where(np.imag(these_poles) > 0, 1, 2))
        axs.scatter(np.real(these_poles), np.imag(these_poles), c=colour_choices, marker=".", s=BIG_SCATTER_STYLE_KWARGS["s"] / 10)

    prepare_axis(axs, None, "Re", "Im", False)
    prepare_figure(fig, None, None, None, None)
    save_figure(fig, "output", file_name)
    fig.show()


def plot_ranking_comparison(rankings, labels, title, file_name : str | None = None):
    fig, axs = plt.subplots()

    im = axs.imshow(len(labels) + 1 - rankings, cmap="coolwarm_r", aspect=2)

    axs.set_yticks(np.arange(len(labels)), labels=labels)
    tick_interval = 5
    axs.set_xticks(np.arange(tick_interval - 1, rankings.shape[1], tick_interval))
    axs.set_xticklabels(np.arange(tick_interval, rankings.shape[1] + 1, tick_interval))

    # Minor ticks
    axs.set_xticks(np.arange(-0.5, rankings.shape[1] + 0.5, 1), minor=True)
    axs.set_yticks(np.arange(-0.5, rankings.shape[0] + 0.5, 1), minor=True)

    # Gridlines based on minor ticks
    axs.grid(which="minor", color="k", linestyle="-", linewidth=0.8)

    prepare_axis(axs, title, "Dataset index", "", False)
    
    cax = axs.inset_axes([1.02, 0, 0.05, 1.], transform=axs.transAxes)
    cbar = fig.colorbar(im, cax=cax, ticks=np.arange(1, len(labels) + 1, 1), orientation="vertical", fraction=0.01)
    cbar.ax.set_yticklabels(np.arange(1, len(labels) + 1, 1)[::-1])

    prepare_figure(fig, "", None, None, None)
    if file_name is not None:
        save_figure(fig, "output", file_name)
    fig.show()


def plot_mse_progression(rmse_results : np.ndarray, noise_values : np.ndarray, model_names : list[str], title : str | None, file_name : str | None):
    fig, axs = plt.subplots()

    for model_ind, model in enumerate(model_names):
        axs.loglog(noise_values, rmse_results[model_ind,:], label=model, c=get_standard_colour(model_ind))

    prepare_axis(axs, None, "$\\sigma_n^2$", "RMSE / $\\sigma_n^2$", True)

    prepare_figure(fig, title, None, None, None)
    if file_name is not None:
        save_figure(fig, "output", file_name)
    fig.show()


def plot_pole_dist(particle_a_means, particle_a_covs, particle_log_weights, ground_truth_coefficients, file_name : str, view_window : list[list[float]] | None  = None):
    fig, axs = plt.subplots()

    def get_roots(coefs):
        return np.roots(np.concatenate([np.ones(1), -coefs]))

    unit_circle = matplotlib.patches.Ellipse(xy=(0, 0), width=2, height=2)
    unit_circle.set_facecolor("#fff")
    unit_circle.set_edgecolor(BLACK)
    unit_circle.set_linewidth(SECONDARY_LINEWIDTH * 0.5)
    unit_circle.set_linestyle(SECONDARY_LINESTYLE)
    unit_circle.set_alpha(1)
    unit_circle.set_fill(False)

    axs.axhline(0, linestyle=SECONDARY_LINESTYLE, linewidth=SECONDARY_LINEWIDTH * 0.5, c=BLACK)
    axs.axvline(0, linestyle=SECONDARY_LINESTYLE, linewidth=SECONDARY_LINEWIDTH * 0.5, c=BLACK)

    axs.add_artist(unit_circle)
    axs.set_aspect("equal")

    NUM_PARTICLES_RENDERING = 50
    weight_order = np.argsort(-particle_log_weights)
    best_particles = weight_order[:NUM_PARTICLES_RENDERING]


    DRAWS_PER_PARTICLE = 20
    process_order = particle_a_means.shape[1]
    colour_centers = np.zeros(process_order, dtype="complex")
    colour_counts = np.zeros(process_order)

    min_x = np.inf 
    max_x = -np.inf
    min_y = np.inf 
    max_y = -np.inf

    for p in list(best_particles):
        processes = np.random.multivariate_normal(mean=particle_a_means[p,:], cov=particle_a_covs[p,:,:], size=(DRAWS_PER_PARTICLE))
        proc_log_weights = scipy.stats.multivariate_normal.logpdf(processes, particle_a_means[p,:], particle_a_covs[p,:,:])
        for i in range(DRAWS_PER_PARTICLE):
            process_poles = get_roots(processes[i, :])
            alpha = np.exp(proc_log_weights[i] + particle_log_weights[p])# - np.max(particle_log_weights))
            if alpha < 1E-2:
                continue
            for j in range(process_order):
                this_pole = process_poles[j].flatten()[0]

                if np.abs(np.imag(this_pole)) < 1E-3:
                    colour_choice = 0
                elif np.imag(this_pole) > 0:
                    colour_choice = 1
                else:
                    colour_choice = 2

                max_x = max(max_x, np.real(this_pole))
                min_x = min(min_x, np.real(this_pole))
                max_y = max(max_y, np.imag(this_pole))
                min_y = min(min_y, np.imag(this_pole))

                axs.scatter(np.real(this_pole), np.imag(this_pole), alpha=max(0, min(1, alpha)), 
                            c=get_standard_colour(colour_choice), 
                            marker=".", s=BIG_SCATTER_STYLE_KWARGS["s"] / 10)

    if ground_truth_coefficients is not None:
        ground_truth_poles = get_roots(ground_truth_coefficients)
        for j in range(process_order):
            this_pole = ground_truth_poles[j].flatten()

            max_x = max(max_x, np.real(this_pole))
            min_x = min(min_x, np.real(this_pole))
            max_y = max(max_y, np.imag(this_pole))
            min_y = min(min_y, np.imag(this_pole))

            axs.scatter(np.real(this_pole), np.imag(this_pole), alpha=1, c=BLACK, 
                        marker="x", label="Ground truth" if j == 0 else None, s=BIG_SCATTER_STYLE_KWARGS["s"] / 4)


    if view_window is None:
        x_diff = max_x - min_x 
        axs.set_xlim([min_x - 0.1 * x_diff, max_x + 0.05 * x_diff])
        
        y_diff = max_y - min_y 
        axs.set_ylim([min_y - 0.1 * y_diff, max_y + 0.05 * y_diff])
    else:
        axs.set_xlim(view_window[0])
        axs.set_ylim(view_window[1])

    prepare_axis(axs, None, "Re", "Im", False)
    prepare_figure(fig, None, None, None, None)
    
    if file_name is not None:
        save_figure(fig, "output", file_name)
    fig.show()


def plot_sigma_estimates(alpha_mean_history, beta_mean_history, ground_truth, file_name):
    fig, axs = plt.subplots()

    err = 0.95
    lower_vals = scipy.stats.invgamma.ppf(0.5 - err / 2, alpha_mean_history, scale=beta_mean_history)
    mean_vals = scipy.stats.invgamma.ppf(0.5, alpha_mean_history, scale=beta_mean_history)
    upper_vals = scipy.stats.invgamma.ppf(0.5 + err / 2, alpha_mean_history, scale=beta_mean_history)

    if ground_truth is not None:
        axs.axhline(ground_truth, c=get_standard_colour(1), label="Ground truth", **SECONDARY_STYLE_KWARGS)
    axs.semilogy(mean_vals, c=BLACK, label="Mean", **PRIMARY_STYLE_KWARGS)
    axs.semilogy(lower_vals, label="95% confidence", **ERROR_BAR_STYLE_KWARGS)
    axs.semilogy(upper_vals, label=None, **ERROR_BAR_STYLE_KWARGS)

    prepare_axis(axs, None, "Time", "$\\sigma^2$", True)
    prepare_figure(fig, "", None, None, None)
    if file_name is not None:
        save_figure(fig, "output", file_name)
    fig.show()


def plot_gaussian_trajectories(ground_truth : np.ndarray, data : data_generation_utils.MeasurementSet, file_name : str, label, means, covs):
    fig, axs = plt.subplots()

    is_first = True
    for clutter in data.get_clutter_history():
        if clutter.size != 1:
            axs.scatter(clutter[:,0], clutter[:,1], label="Clutter" if is_first else None, **CLUTTER_SCATTER_STYLE_KWARGS)
            is_first = False


    axs.plot(means[:,0], means[:,1], c=get_standard_colour(1), label=label, **PRIMARY_STYLE_KWARGS)


    axs.plot(ground_truth[:,0], ground_truth[:,1], label="Ground truth", c=BLACK, **SECONDARY_STYLE_KWARGS)
    

    is_first = True
    for meas in data.get_uncluttered_data():
        if meas.size != 1:
            axs.scatter(meas[:,0], meas[:,1], label="True measurement" if is_first else None, c=BLACK, **PRIMARY_SCATTER_STYLE_KWARGS)
            is_first = False 

    axs.scatter(ground_truth[0,0], ground_truth[0,1], marker="o", c=BLACK, s=100, zorder=100)
    axs.scatter(ground_truth[-1,0], ground_truth[-1,1], marker="x", c=BLACK, s=100, zorder=100)

    prepare_axis(axs, "Tracking trajectories", "x", "y", True)
    axs.legend(loc="upper left")
    prepare_figure(fig, "", None, None, None)

    if file_name is not None:
        save_figure(fig, "output", file_name)
    fig.show()


def plot_particle_trajectories(ground_truth, data, file_name, label, particle_history : list[ParticleCloud], zn_mean_history):
    fig, axs = plt.subplots()

    is_first = True
    for clutter in data.get_clutter_history():
        if clutter.size != 1:
            axs.scatter(clutter[:,0], clutter[:,1], label="Clutter" if is_first else None, **CLUTTER_SCATTER_STYLE_KWARGS)
            is_first = False
        
    start_ind = 0
    for pc in particle_history:
        if pc is not None:
            num_particles = pc.num_particles
            break 
        start_ind += 1

    full_zn_history = np.zeros((len(particle_history), num_particles, 2))
    log_weight_history = np.zeros((len(particle_history), num_particles))


    for t, pc in enumerate(particle_history):
        if t < start_ind:
            continue
        full_zn_history[t,:,:] = pc.Z[:,:,0]
        log_weight_history[t,:] = pc.log_weights

    max_weights = np.max(log_weight_history, axis=1)

    num_plotted = np.zeros(len(particle_history))

    for p in range(num_particles):
        for t in range(start_ind + 1, full_zn_history.shape[0]):
            if log_weight_history[t,p] - max_weights[t] < np.log(0.5) or num_plotted[t] > 100:
                continue
            num_plotted[t] += 1
            axs.plot(full_zn_history[t-1:t+1,p,0], full_zn_history[t-1:t+1,p,1], 
                     alpha=0.05 * np.exp(log_weight_history[t,p] - max_weights[t]), c=get_standard_colour(1), **SECONDARY_STYLE_KWARGS, zorder=0)

    axs.plot(zn_mean_history[:,0], zn_mean_history[:,1], label=label, c=RED, **PRIMARY_STYLE_KWARGS, zorder=5)
    axs.plot(ground_truth[:,0], ground_truth[:,1], label="Ground truth", c=BLACK, **SECONDARY_STYLE_KWARGS, zorder=10)


    axs.scatter(ground_truth[0,0], ground_truth[0,1], marker="o", c=BLACK, s=100, zorder=100)
    axs.scatter(ground_truth[-1,0], ground_truth[-1,1], marker="x", c=BLACK, s=100, zorder=100)

    is_first = True
    for meas in data.get_uncluttered_data():
        if meas.size != 1:
            axs.scatter(meas[:,0], meas[:,1], label="True measurement" if is_first else None, c=BLACK, **PRIMARY_SCATTER_STYLE_KWARGS)
            is_first = False 

    prepare_axis(axs, "Tracking trajectories", "x", "y", True)
    prepare_figure(fig, "", None, None, None)

    if file_name is not None:
        save_figure(fig, "output", file_name)
    fig.show()


def plot_eff_sample_size(effective_num_particles_history : np.ndarray, params : settings.TrackerParameters, file_name : str):
    fig, axs = plt.subplots(nrows=1, ncols=1)
    axs.plot(np.arange(params.model_order, effective_num_particles_history.size), effective_num_particles_history[params.model_order:], c=get_standard_colour(0), **PRIMARY_STYLE_KWARGS)
    axs.axhline(params.num_particles, c=get_standard_colour(2), **SECONDARY_STYLE_KWARGS, label="Num particles")
    axs.axhline(params.min_particles, c=get_standard_colour(1), **SECONDARY_STYLE_KWARGS, label="Resampling threshold")
    axs.set_ylim([0, params.num_particles * 1.1])
    axs = prepare_axis(axs, "Effective particles", "Time", "Num particles", True)
    prepare_figure(fig, "", None, None, None)
    save_figure(fig, "figs", file_name)
    fig.show()


def plot_particle_rmse_history(particle_history, ground_truth_history, burn_in):
    fig, axs = plt.subplots(nrows=1, ncols=1)
    rmse_history = pf_utils.get_rmse_history(ground_truth_history, particle_history, burn_in)

    axs.plot(np.arange(burn_in, ground_truth_history.shape[0]), rmse_history)

    axs = prepare_axis(axs, "RMSE history", "Time", "RMSE", False)
    prepare_figure(fig, "", None, None, None)
    fig.show()


def export_result_table(data : np.ndarray, file_name : str):
    df = pd.DataFrame(data)
    df.to_csv(f"output/{file_name}.csv", index=False, header=False, lineterminator="\n", float_format="%.3g")