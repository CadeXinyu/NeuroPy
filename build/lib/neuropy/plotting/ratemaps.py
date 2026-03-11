import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from .. import core
from ..utils import mathutil
from .figure import Fig


def plot_ratemap(
    ratemap: core.Ratemap,
    normalize_xbin=False,
    ax=None,
    pad=2,
    normalize_tuning_curve=False,
    cross_norm=None,
    sortby=None,
    sortby_cellIDs=None,
    return_cellIDs=False,
    cmap="tab20b",
):
    """Plot 1D place fields stacked

    Parameters
    ----------
    ax : [type], optional
        [description], by default None
    speed_thresh : bool, optional
        [description], by default False
    pad : int, optional
        [description], by default 2
    normalize_xbin : bool, optional
        [description], by default False
    normalize_tuning_curve : bool, optional
        [description], by default False
    cross_norm : np.array, optional
        Nx2 numpy array including xmin and xptp per neuron, by default None.
    sortby : array, optional #send in sort 
        [description], by default None
    sortby_cellIDs : array, optional
        Array of cell IDs to sort by, this will override the sortby parameter if provided. 
        This is for sorting by cellIDs rather than indices.
    cmap : str, optional
        [description], by default "tab20b"

    Returns
    -------
    [type]
        [description]
    """
    cmap = mpl.cm.get_cmap(cmap)

    tuning_curves = ratemap.tuning_curves
    n_neurons = ratemap.n_neurons
    neuron_ids = ratemap.neuron_ids 
    # bin_cntr = ratemap.xbin_centers
    bin_cntr = ratemap.x_coords()
    if normalize_xbin:
        bin_cntr = (bin_cntr - np.min(bin_cntr)) / np.ptp(bin_cntr)

    if ax is None:
        fig = Fig(nrows=1, ncols=1, size=(4.5, 11))
        ax = fig.subplot(fig.gs[0])

    if normalize_tuning_curve:
        if isinstance(cross_norm, np.ndarray):
            # Create xmin array and set it to be broadcastable during
            xmin = cross_norm[:,0]
            xptp = cross_norm[:,1] - cross_norm[:,0]

            tuning_curves = mathutil.min_max_external_scaler(tuning_curves, xmin, xptp)
        else:
            tuning_curves = mathutil.min_max_scaler(tuning_curves)
        pad = 1

    # ------- TMP RW 4/3/25 -------------------- #
    # if sortby is None:
    #     sort_ind = np.argsort(np.argmax(tuning_curves, axis=1))
    # elif isinstance(sortby, (list, np.ndarray)):
    #     sort_ind = sortby
    # else:
    #     sort_ind = np.arange(n_neurons)

    # if isinstance(sortby_cellIDs ,(list,np.ndarray)):
    #    # if len(sort_ind) != len(tuning_curves): #if there are gaps
    #     sortby = sortby_cellIDs
    #     neuron_ids = ratemap.neuron_ids
    #     tuning_curve_sorted = np.zeros((len(sort_ind),tuning_curves.shape[1]))  # Fill gaps with zeros
    #     #if missing
    #     for i, neuron in enumerate(sort_ind):
    #         if neuron in neuron_ids:
    #             index = np.where(neuron_ids == neuron)[0][0]  # Find index in B
    #             tuning_curve_sorted[i] = tuning_curve_sorted[index]  # Assign existing tuning curve
    #     #if extra
    #     extra = [n for n in neuron_ids if n not in sort_ind]  # Find extra B neurons
    #     extra_tuning_curves = np.array([tuning_curves[np.where(neuron_ids == n)[0][0]] for n in extra])
    #     sorted_neurons = np.concatenate([sorted_neurons, extra])  # Append IDs
    #     tuning_curve_sorted= np.vstack([tuning_curve_sorted, extra_tuning_curves])  # Append tuning curves

    #     tuning_curves = tuning_curve_sorted
    # ------- END TMP RW 4/3/25 -------------------- #
    

    
    if sortby_cellIDs is not None and isinstance(sortby_cellIDs, (list, np.ndarray)):
        # Sort by cell IDs if provided
        neuron_ids = ratemap.neuron_ids
        tuning_curve_sorted = np.zeros((len(sortby_cellIDs), tuning_curves.shape[1]))  # Fill gaps with zeros

        # Handle missing and extra neurons
        for i, neuron in enumerate(sortby_cellIDs):
            if neuron in neuron_ids:
                index = np.where(neuron_ids == neuron)[0][0]  # Find index in neuron_ids
                tuning_curve_sorted[i] = tuning_curves[index]  # Assign existing tuning curve
        # Find extra neurons not in sortby_cellIDs
        extra = [n for n in neuron_ids if n not in sortby_cellIDs]
        extra_indices = [np.where(neuron_ids == n)[0][0] for n in extra]
        extra_tuning_curves = tuning_curves[extra_indices]

        # Sort the extra neurons by np.argsort(np.argmax(...))
        extra_sort_indices = np.argsort(np.argmax(extra_tuning_curves, axis=1))
        extra = np.array(extra)[extra_sort_indices]  # Sort extra neuron IDs
        extra_tuning_curves = extra_tuning_curves[extra_sort_indices]  # Sort extra tuning curves

        # Append the sorted extra neurons and their tuning curves
        sortby_cellIDs = np.concatenate([sortby_cellIDs, extra])  # Append sorted extra neuron IDs
        tuning_curve_sorted = np.vstack([tuning_curve_sorted, extra_tuning_curves])  # Append sorted extra tuning curves

        # Update tuning_curves and sort indices
        tuning_curves = tuning_curve_sorted
        sort_ind = np.arange(len(sortby_cellIDs))  # Update sort indices
        neuron_ids = sortby_cellIDs       
    else:
        # Original behavior: sort by indices or default
        if sortby is None:
            sort_ind = np.argsort(np.argmax(tuning_curves, axis=1))
        elif isinstance(sortby, (list, np.ndarray)):
            sort_ind = sortby
        else:
            sort_ind = np.arange(n_neurons) 
            
     # Define the colormap
    #cmap = plt.cm.viridis  # Use any colormap you prefer
    #extra_color = "gray"  # Color for extra neurons

        # Compute the colors for all neurons
    colors = [cmap(i / len(sort_ind)) for i in range(len(sort_ind))]
    
    # if isinstance(extra, (list, np.ndarray)) and len(extra) > 0:
    #     #make last len(extra) in colors be gray
    #     colors[-len(extra):] = ["gray"] * len(extra)
    
    # Plotting loop
    for i, neuron_ind in enumerate(sort_ind):
        color = colors[i]  # Use the precomputed color       
      #  color = cmap(i / len(sort_ind))

        # Check if the tuning curve is all zeros
        if np.all(tuning_curves[neuron_ind] == 0):
            # Skip plotting the filled area and line, but leave space for the blank plot
            continue

        ax.fill_between(
            bin_cntr,
            i * pad,
            i * pad + tuning_curves[neuron_ind],
            color=color,
            ec=None,
            alpha=0.7,
            zorder=i + 1,
        )
        ax.plot(
            bin_cntr,
            i * pad + tuning_curves[neuron_ind],
            color=color,
            alpha=1,
            lw=0.6,
        )

    ax.set_yticks(list(range(len(sort_ind))))
  #  ax.set_yticklabels(list(ratemap.neuron_ids[sort_ind]))
    ax.set_yticklabels(list(neuron_ids[sort_ind]))
    ax.set_xlabel("Position")
    ax.spines["left"].set_visible(False)
    if normalize_xbin:
        ax.set_xlim([0, 1])
    ax.tick_params("y", length=0)
    ax.set_ylim([0, len(sort_ind)])
    # if self.run_dir is not None:
    #     ax.set_title(self.run_dir.capitalize() + " Runs only")

    if return_cellIDs:
        return ax, neuron_ids[sort_ind]
    else:
        return ax


def plot_raw(self, ax=None, subplots=(8, 9)):
    """Plot spike location on animal's path

    Parameters
    ----------
    speed_thresh : bool, optional
        [description], by default False
    ax : [type], optional
        [description], by default None
    subplots : tuple, optional
        [description], by default (8, 9)
    """

    mapinfo = self.ratemaps
    nCells = len(mapinfo["pos"])

    def plot_(cell, ax):
        if subplots is None:
            ax.clear()
        ax.plot(self.x, self.t, color="gray", alpha=0.6)
        ax.plot(mapinfo["pos"][cell], mapinfo["spikes"][cell], ".", color="#ff5f5c")
        ax.set_title(
            " ".join(filter(None, ("Cell", str(cell), self.run_dir.capitalize())))
        )
        ax.invert_yaxis()
        ax.set_xlabel("Position (cm)")
        ax.set_ylabel("Time (s)")

    if ax is None:

        if subplots is None:
            _, gs = Fig().draw(grid=(1, 1), size=(6, 8))
            ax = plt.subplot(gs[0])
            widgets.interact(
                plot_,
                cell=widgets.IntSlider(
                    min=0,
                    max=nCells - 1,
                    step=1,
                    description="Cell ID:",
                ),
                ax=widgets.fixed(ax),
            )
        else:
            _, gs = Fig().draw(grid=subplots, size=(10, 11))
            for cell in range(nCells):
                ax = plt.subplot(gs[cell])
                ax.set_yticks([])
                plot_(cell, ax)
