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
    exclude_extra=False,
    return_cellIDs=False,
    cmap="tab20b",
    sort_method="peak",  # <--- NEW PARAMETER
):
    """Plot 1D place fields stacked vertically.

    Parameters
    ----------
    ...
    sort_method : str, optional
        Method to sort neurons if sortby is None. Options: "peak", "com" (default: "peak")
    """
    # ... [Existing setup code remains the same] ...
    
    cmap = mpl.cm.get_cmap(cmap)
    tuning_curves = ratemap.tuning_curves
    n_neurons = ratemap.n_neurons
    neuron_ids = ratemap.neuron_ids
    bin_cntr = ratemap.x_coords()

    if normalize_xbin:
        bin_cntr = (bin_cntr - np.min(bin_cntr)) / np.ptp(bin_cntr)

    if ax is None:
        fig = Fig(nrows=1, ncols=1, size=(4.5, 11))
        ax = fig.subplot(fig.gs[0])

    if normalize_tuning_curve:
        if isinstance(cross_norm, np.ndarray):
            xmin = cross_norm[:, 0]
            xptp = cross_norm[:, 1] - cross_norm[:, 0]
            tuning_curves = mathutil.min_max_external_scaler(tuning_curves, xmin, xptp)
        else:
            tuning_curves = mathutil.min_max_scaler(tuning_curves)
        pad = 1

    # Handle sorting
    if sortby_cellIDs is not None and isinstance(sortby_cellIDs, (list, np.ndarray)):
        tuning_curves, sort_ind, neuron_ids = _sort_by_cell_ids(
            sortby_cellIDs, neuron_ids, tuning_curves, exclude_extra
        )
    else:
        # Pass the new sort_method down to the helper
        sort_ind = _get_sort_indices(sortby, tuning_curves, n_neurons, method=sort_method)

    # ... [Rest of the plotting code remains exactly the same] ...
    
    colors = [cmap(i / len(sort_ind)) for i in range(len(sort_ind))]

    for i, neuron_ind in enumerate(sort_ind):
        if np.all(tuning_curves[neuron_ind] == 0):
            continue

        ax.fill_between(
            bin_cntr,
            i * pad,
            i * pad + tuning_curves[neuron_ind],
            color=colors[i],
            ec=None,
            alpha=0.7,
            zorder=i + 1,
        )
        ax.plot(
            bin_cntr,
            i * pad + tuning_curves[neuron_ind],
            color=colors[i],
            alpha=1,
            lw=0.6,
        )

    ax.set_yticks(list(range(len(sort_ind))))
    ax.set_yticklabels(list(neuron_ids[sort_ind]))
    ax.set_xlabel("Position")
    ax.spines["left"].set_visible(False)
    ax.tick_params("y", length=0)
    ax.set_ylim([0, len(sort_ind)])
    
    if normalize_xbin:
        ax.set_xlim([0, 1])

    if return_cellIDs:
        return ax, neuron_ids[sort_ind]
    else:
        return ax

def _sort_by_cell_ids(sortby_cellIDs, neuron_ids, tuning_curves, exclude_extra=False):
    """Sort tuning curves by specified cell IDs.
    
    Neurons in sortby_cellIDs are placed first in specified order.
    By default, extra neurons not in sortby_cellIDs are appended, sorted by peak position.
    Missing neurons are represented with zero-filled tuning curves.
    
    Parameters
    ----------
    sortby_cellIDs : array-like
        Cell IDs in desired order
    neuron_ids : np.ndarray
        Available neuron IDs in ratemap
    tuning_curves : np.ndarray
        Tuning curves for all neurons
    exclude_extra : bool, optional
        If True, exclude neurons not in sortby_cellIDs (default: False)
        
    Returns
    -------
    tuning_curves_sorted : np.ndarray
        Sorted tuning curves
    sort_ind : np.ndarray
        Sorting indices
    neuron_ids_sorted : np.ndarray
        Sorted neuron IDs
    """
    n_bins = tuning_curves.shape[1]
    tuning_curves_sorted = np.zeros((len(sortby_cellIDs), n_bins))

    # Map requested cell IDs to their tuning curves
    for i, neuron in enumerate(sortby_cellIDs):
        if neuron in neuron_ids:
            idx = np.where(neuron_ids == neuron)[0][0]
            tuning_curves_sorted[i] = tuning_curves[idx]

    # Handle extra neurons (those not in sortby_cellIDs)
    if not exclude_extra:
        # Find neurons not in sortby_cellIDs
        extra_neurons = [n for n in neuron_ids if n not in sortby_cellIDs]
        
        if extra_neurons:
            # Get tuning curves for extra neurons
            extra_indices = [np.where(neuron_ids == n)[0][0] for n in extra_neurons]
            extra_tuning_curves = tuning_curves[extra_indices]

            # Sort extra neurons by peak position
            extra_sort_idx = np.argsort(np.argmax(extra_tuning_curves, axis=1))
            extra_neurons_sorted = np.array(extra_neurons)[extra_sort_idx]
            extra_tuning_curves_sorted = extra_tuning_curves[extra_sort_idx]

            # Append sorted extra neurons
            sortby_cellIDs = np.concatenate([sortby_cellIDs, extra_neurons_sorted])
            tuning_curves_sorted = np.vstack([tuning_curves_sorted, extra_tuning_curves_sorted])

    # Create sequential sort indices
    sort_ind = np.arange(len(sortby_cellIDs))
    
    return tuning_curves_sorted, sort_ind, sortby_cellIDs


def _get_sort_indices(sortby, tuning_curves, n_neurons, method="peak"):
    """Get sorting indices based on sortby parameter or method.
    
    Parameters
    ----------
    sortby : array-like or None
        Sorting indices or None for automatic sorting
    tuning_curves : np.ndarray
        Tuning curves (n_neurons, n_bins)
    n_neurons : int
        Number of neurons
    method : str
        "peak" for argmax, "com" for center of mass
        
    Returns
    -------
    sort_ind : np.ndarray
        Sorting indices
    """
    # 1. User provided explicit sort indices (Manual Override)
    if isinstance(sortby, (list, np.ndarray)):
        return sortby

    # 2. Sort by Center of Mass (Weighted Average)
    if method == "com":
        n_bins = tuning_curves.shape[1]
        x_grid = np.arange(n_bins)
        
        # Sum of rates (mass) per neuron
        total_mass = np.sum(tuning_curves, axis=1)
        
        # Handle silent cells (avoid divide by zero)
        # We set mass to 1 temporarily; the numerator will be 0 anyway, resulting in COM=0
        safe_mass = total_mass.copy()
        safe_mass[safe_mass == 0] = 1
        
        # Sum of (rate * position)
        weighted_pos = np.sum(tuning_curves * x_grid, axis=1)
        
        com = weighted_pos / safe_mass
        
        # Optional: Push silent cells (total_mass=0) to the end
        # com[total_mass == 0] = 99999 
        
        return np.argsort(com)

    # 3. Default: Sort by Peak Location (argmax)
    else:
        return np.argsort(np.argmax(tuning_curves, axis=1))


def plot_raw(self, ax=None, subplots=(8, 9)):
    """Plot spike locations overlaid on animal's trajectory.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None with subplots=None, creates interactive widget (default: None)
    subplots : tuple, optional
        (rows, cols) for subplot grid. If None, creates interactive widget (default: (8, 9))
    """
    mapinfo = self.ratemaps
    nCells = len(mapinfo["pos"])

    def plot_(cell, ax):
        """Helper function to plot single cell."""
        if subplots is None:
            ax.clear()
            
        # Plot trajectory in gray
        ax.plot(self.x, self.t, color="gray", alpha=0.6)
        
        # Plot spike positions in red
        ax.plot(mapinfo["pos"][cell], mapinfo["spikes"][cell], ".", color="#ff5f5c")
        
        # Configure plot appearance
        title_parts = ["Cell", str(cell)]
        if hasattr(self, 'run_dir') and self.run_dir:
            title_parts.append(self.run_dir.capitalize())
        ax.set_title(" ".join(title_parts))
        
        ax.invert_yaxis()
        ax.set_xlabel("Position (cm)")
        ax.set_ylabel("Time (s)")

    if ax is None:
        if subplots is None:
            # Create interactive widget for single cell view
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
            # Create grid of subplots showing all cells
            _, gs = Fig().draw(grid=subplots, size=(10, 11))
            for cell in range(nCells):
                ax = plt.subplot(gs[cell])
                ax.set_yticks([])
                plot_(cell, ax)