from dataclasses import dataclass

import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from tqdm import tqdm
from scipy.signal import find_peaks, peak_widths
from copy import deepcopy
import seaborn as sns

from neuropy import core
from neuropy.utils.signal_process import ThetaParams
from neuropy import plotting
from neuropy.utils.mathutil import contiguous_regions
from neuropy.externals.peak_prominence2d import getProminence
from neuropy.utils.circular import circular_linear_regression_cuda, circular_linear_regression
from neuropy.utils.circular import interpolate_phase_circular
from ipywidgets import interact, IntSlider


class Pf1D(core.Ratemap):
    
    def __init__(
        self,
        neurons: core.Neurons,
        position: core.Position,
        epochs: core.Epoch = None,
        frate_thresh: float = None,
        speed_thresh: float = None,
        spkcount_thresh: int = 0,
        grid_bin: int = None,
        bin_num: int = None,
        sigma_bin: float = 0,
        sigma_pos: float = 0,
        mode: str = "linear",
        verbose: bool = True,
    ):
        # Validate inputs
        assert position.ndim == 1, "Only 1 dimensional position are acceptable"
        assert (grid_bin is None) != (bin_num is None), \
            "Exactly one of 'grid_bin' or 'bin_num' must be provided (not both or neither)"
        
        # Extract basic parameters
        neuron_ids = neurons.neuron_ids
        position_srate = position.sampling_rate
        
        # Smooth position if requested
        if sigma_pos > 0:
            position = position.get_smoothed(sigma_pos)
        
        x = position.x
        speed = position.speed
        t = position.time
        t_start = position.t_start
        t_stop = position.t_stop

        # Validate mode
        if mode not in ("linear", "circular"):
            raise ValueError(f"mode must be 'linear' or 'circular', got '{mode}'")

        # Create spatial bins
        self.mode = mode
        if self.mode == 'circular':
            x_min, x_max = 0, 2 * np.pi
        else:
            x_min, x_max = np.nanmin(x), np.nanmax(x)
        x_range = x_max - x_min
        
        if bin_num is not None:
            n_bins = bin_num
            xbin = np.linspace(x_min, x_max, n_bins + 1)
            grid_bin = x_range / n_bins
        else:
            n_bins = int(np.ceil(x_range / grid_bin))
            xbin = np.linspace(x_min, x_max, n_bins + 1)
        
        self.grid_bin = grid_bin
        
        # Define smoothing function (with explicit float casting to prevent integer truncation)
        if mode == 'circular' and sigma_bin > 0:
            def smooth_(f):
                f = np.asarray(f, dtype=float)
                is_1d = f.ndim == 1
                if is_1d:
                    f = f.reshape(1, -1)
                
                smoothed = core.Ratemap._circular_gaussian_smooth(
                    f, sigma_bin/grid_bin
                )
                
                if is_1d:
                    smoothed = smoothed.squeeze()
                return smoothed
        else:
            from scipy.ndimage import gaussian_filter1d
            smooth_ = lambda f: (
                gaussian_filter1d(np.asarray(f, dtype=float), sigma_bin / grid_bin, axis=-1) 
                if sigma_bin > 0 else np.asarray(f, dtype=float)
            )
        
        # Extract spikes and position based on epochs or speed threshold
        if epochs is not None:
            assert isinstance(epochs, core.Epoch), "epochs should be core.Epoch object"
            
            spiketrains = [
                np.concatenate([
                    spktrn[(spktrn >= epc.start) & (spktrn <= epc.stop)]
                    for epc in epochs.to_dataframe().itertuples()
                ])
                for spktrn in neurons.spiketrains
            ]
            
            indx = np.concatenate([
                np.where((t >= epc.start) & (t <= epc.stop))[0]
                for epc in epochs.to_dataframe().itertuples()
            ])
            
            # --- FIX 1: Apply speed threshold to epoch occupancy indices ---
            if speed_thresh is not None:
                valid_speed_mask = speed[indx] >= speed_thresh
                indx = indx[valid_speed_mask]
            
            if verbose:
                print("Note: speed_thresh is correctly applied to BOTH spikes and epoch occupancy.")
        else:
            spiketrains = neurons.time_slice(t_start, t_stop).spiketrains
            indx = np.where(speed >= speed_thresh)[0]
        
        x_thresh = x[indx]
        
        # Compute spike positions and counts
        spk_pos, spk_t, spkcounts = [], [], []
        for spktrn in spiketrains:
            spk_spd = np.interp(spktrn, t, speed)

            # --- FIX 2: Use nearest-neighbor matching for circular tracks ---
            if mode == 'circular' and x_range > 0:
                # Find indices of nearest neighbors in time (t)
                indices = np.searchsorted(t, spktrn)
                indices = np.clip(indices, 0, len(t) - 1)
                prev_indices = np.clip(indices - 1, 0, len(t) - 1)
                
                dist = np.abs(t[indices] - spktrn)
                dist_prev = np.abs(t[prev_indices] - spktrn)
                
                # Choose the index that is closer in time
                closer_indx = np.where(dist < dist_prev, indices, prev_indices)
                spk_x = x[closer_indx]  # Use raw discontinuous x trace directly
            else:
                spk_x = np.interp(spktrn, t, x)
            
            # Apply speed threshold to spikes
            if speed_thresh is not None:
                speed_mask = np.where(spk_spd >= speed_thresh)[0]
                spk_x = spk_x[speed_mask]
                spktrn = spktrn[speed_mask]
            
            # --- FIX 3: Strict filtering of NaNs and out-of-bounds values ---
            valid_mask = ~np.isnan(spk_x) & (spk_x >= xbin[0]) & (spk_x <= xbin[-1])
            spk_x = spk_x[valid_mask]
            spktrn = spktrn[valid_mask]
            
            spk_pos.append(spk_x)
            spk_t.append(spktrn)
            spkcounts.append(np.histogram(spk_x, bins=xbin)[0])
        
        # --- FIX 4 & 5: Robust firing rate map calculation with Occupancy Masking ---
        spkcounts = smooth_(np.asarray(spkcounts, dtype=float))
        
        # Calculate raw occupancy WITHOUT the 1e-16 hack
        raw_occupancy = np.histogram(x_thresh, bins=xbin)[0] / position_srate
        occupancy = smooth_(np.asarray(raw_occupancy, dtype=float))
        
        # Initialize tuning curve with zeros
        tuning_curve = np.zeros_like(spkcounts, dtype=float)
        
        # Create an occupancy mask (e.g., minimum 0.001 seconds spent in a bin to be considered valid)
        min_occ_thresh = 0.001
        valid_bins = occupancy > min_occ_thresh
        
        # Only compute firing rate for valid, visited bins
        tuning_curve[:, valid_bins] = spkcounts[:, valid_bins] / occupancy[valid_bins]
        
        # Filter by peak firing rate threshold
        frate_thresh_indx = np.where(np.max(tuning_curve, axis=1) >= frate_thresh)[0]
        tuning_curve = tuning_curve[frate_thresh_indx, :]
        neuron_ids = neuron_ids[frate_thresh_indx]
        spk_t = [spk_t[i] for i in frate_thresh_indx]
        spk_pos = [spk_pos[i] for i in frate_thresh_indx]
        
        # Filter by minimum spike count
        n_spikes = np.array([len(spikes) for spikes in spk_t])
        spk_thresh_indx = np.where(n_spikes >= spkcount_thresh)[0]
        tuning_curve = tuning_curve[spk_thresh_indx, :]
        neuron_ids = neuron_ids[spk_thresh_indx]
        spk_t = [spk_t[i] for i in spk_thresh_indx]
        spk_pos = [spk_pos[i] for i in spk_thresh_indx]
        n_spikes = n_spikes[spk_thresh_indx]
        
        # Initialize parent class
        super().__init__(
            tuning_curves=tuning_curve, 
            coords=(xbin[:-1] + xbin[1:]) / 2, 
            neuron_ids=neuron_ids
        )
        
        # Store attributes
        self.ratemap_spiketrains = spk_t
        self.ratemap_spiketrains_pos = spk_pos
        self.n_spikes = n_spikes
        self.occupancy = occupancy
        self.frate_thresh = frate_thresh
        self.speed_thresh = speed_thresh
        self.spkcount_thresh = spkcount_thresh
        self.speed = speed
        self.t = t
        self.x = x
        self.x_thresh = x_thresh
        self.xbin = xbin
        self.t_start = t_start
        self.t_stop = t_stop
        self.sigma_bin = sigma_bin
        self.sigma_pos = sigma_pos

    
    def __repr__(self):
        """String representation showing key information about the Pf1D object."""
        lines = []
        lines.append("=" * 70)
        lines.append("Pf1D: 1D Place Field Analysis")
        lines.append("=" * 70)
        
        # Basic parameters
        lines.append(f"Neurons: {len(self.neuron_ids)}")
        lines.append(f"Bins: {len(self.coords)}, "
                    f"size={np.diff(self.xbin)[0]:.2f}, "
                    f"range=[{np.min(self.xbin):.2f}, {np.max(self.xbin):.2f}]")
        lines.append(f"Duration: {self.t_stop - self.t_start:.2f}s "
                    f"({(self.t_stop - self.t_start)/60:.2f}min)")
        
        # Filter settings
        filter_parts = [f"frate≥{self.frate_thresh}Hz", 
                    f"spikes≥{self.spkcount_thresh}"]
        if self.speed_thresh is not None:
            filter_parts.append(f"speed≥{self.speed_thresh}")
        lines.append(f"Filters: {', '.join(filter_parts)}")
        
        # Firing statistics
        spike_counts = [len(spikes) for spikes in self.ratemap_spiketrains]
        if len(spike_counts) > 0:
            peak_rates = np.max(self.tuning_curves, axis=1)
            lines.append(f"Firing: {sum(spike_counts)} spikes, "
                        f"peak rate={np.mean(peak_rates):.2f}±{np.std(peak_rates):.2f}Hz")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def shuffle_tc(self, n_shuffles=1000, random_seed=None, verbose=True):
        """
        Calculate spatial information and p-values using shuffle testing.
        
        Formula for spatial information:
        SI = Σ p_i * (r_i / r̄) * log2(r_i / r̄)
        
        where:
        - p_i is the probability of being in the ith bin (time spent / total time)
        - r_i is the firing rate in the ith bin
        - r̄ is the overall mean firing rate
        
        Args:
            n_shuffles: Number of shuffle iterations (default: 1000)
            random_seed: Random seed for reproducibility (default: None)
            verbose: Print progress information (default: True)
        
        Returns:
            pd.DataFrame with columns:
                - neuron_id: neuron identifier
                - spatial_info: spatial information in bits/event
                - pvalue: p-value from shuffle test
                - n_spikes: total spike count
                - mean_rate: overall mean firing rate (Hz)
                - peak_rate: peak firing rate (Hz)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        n_neurons = len(self.neuron_ids)
        
        # Pre-compute occupancy probability
        total_time = np.sum(self.occupancy)
        p_i = self.occupancy / total_time
        
        # Initialize results
        results = []
        
        if verbose:
            print(f"Calculating spatial information and p-values for {n_neurons} neurons...")
            iterator = tqdm(range(n_neurons), desc="Processing neurons")
        else:
            iterator = range(n_neurons)
        
        for neuron_idx in iterator:
            spk_positions = self.ratemap_spiketrains_pos[neuron_idx]
            n_spikes = len(spk_positions)
            
            if n_spikes == 0:
                results.append({
                    'neuron_id': self.neuron_ids[neuron_idx],
                    'spatial_info': 0.0,
                    'pvalue': 1.0,
                    'n_spikes': 0,
                    'mean_rate': 0.0,
                    'peak_rate': 0.0
                })
                continue
            
            # Get firing rate from tuning curve
            r_i = self.tuning_curves[neuron_idx, :]
            r_mean = n_spikes / total_time
            r_peak = np.max(r_i)
            
            # Calculate spatial information (bits per event)
            mask = r_i > 0
            if np.sum(mask) > 0:
                si = np.sum(
                    p_i[mask] * (r_i[mask] / r_mean) * np.log2(r_i[mask] / r_mean)
                )
            else:
                si = 0.0
            
            # Perform shuffle test
            shuffle_si = np.zeros(n_shuffles)
            for shuffle_idx in range(n_shuffles):
                # Randomly shuffle spike positions
                shuffled_positions = np.random.choice(
                    self.x_thresh,
                    size=n_spikes,
                    replace=True
                )
                
                # Calculate spike counts for shuffled data
                shuffled_counts, _ = np.histogram(shuffled_positions, bins=self.xbin)
                r_i_shuffle = shuffled_counts / self.occupancy
                
                # Calculate spatial information for shuffle
                mask_shuffle = r_i_shuffle > 0
                if np.sum(mask_shuffle) > 0:
                    shuffle_si[shuffle_idx] = np.sum(
                        p_i[mask_shuffle] * (r_i_shuffle[mask_shuffle] / r_mean) * 
                        np.log2(r_i_shuffle[mask_shuffle] / r_mean)
                    )
            
            # Calculate p-value
            pvalue = np.sum(shuffle_si >= si) / n_shuffles
            
            results.append({
                'neuron_id': self.neuron_ids[neuron_idx],
                'spatial_info': si,
                'pvalue': pvalue,
                'n_spikes': n_spikes,
                'mean_rate': r_mean,
                'peak_rate': r_peak
            })
        
        df = pd.DataFrame(results)
        
        if verbose:
            n_significant = np.sum(df['pvalue'] < 0.05)
            print(f"\nSummary:")
            print(f"  Total neurons: {n_neurons}")
            print(f"  Significant place cells (p < 0.05): {n_significant} ({100*n_significant/n_neurons:.1f}%)")
            print(f"  Mean spatial information: {df['spatial_info'].mean():.4f} bits/event")
            print(f"  Mean spatial information (significant): {df[df['pvalue'] < 0.05]['spatial_info'].mean():.4f} bits/event" if n_significant > 0 else "")
        
        return df
    
    def get_extrema_interpolated_phase(self, trace):
        """
        Calculates phase based on linear interpolation between local maxima and minima.
        
        Rules:
        - Min to Max: Phase linearly increases from 0 to pi
        - Max to Min: Phase linearly increases from pi to 2pi
        """
        trace -= np.median(trace)
        
        # 1. Identify Zero Crossings
        # We look for points where the sign of the trace changes
        # np.signbit is generally faster and handles 0s consistently
        zero_crossings = np.where(np.diff(np.signbit(trace)))[0]
        
        extrema_indices = []
        extrema_types = []  # Stores 'max' or 'min'
        
        # 2. Find Max/Min between each pair of zero crossings
        for i in range(len(zero_crossings) - 1):
            # Define the interval between two zero crossings
            # We start at the crossing + 1 to avoid overlap, but include the next crossing
            idx_start = zero_crossings[i] + 1
            idx_end = zero_crossings[i+1] + 1
            
            # Safety check for empty slice
            if idx_start >= idx_end:
                continue
                
            segment = trace[idx_start:idx_end]
            segment_indices = np.arange(idx_start, idx_end)
            
            # Check if this segment is above or below zero
            # (We use the mean to be robust against noise near zero)
            if np.mean(segment) > 0:
                # Positive segment -> Find Max
                local_max_idx = segment_indices[np.argmax(segment)]
                extrema_indices.append(local_max_idx)
                extrema_types.append('max')
            else:
                # Negative segment -> Find Min
                local_min_idx = segment_indices[np.argmin(segment)]
                extrema_indices.append(local_min_idx)
                extrema_types.append('min')
                
        # Initialize phase array with NaNs (for points outside the first/last extrema)
        phase_interpolated = np.full(trace.shape, np.nan)
        
        # 3. Linear Interpolate Phase between Extrema
        for i in range(len(extrema_indices) - 1):
            curr_idx = extrema_indices[i]
            next_idx = extrema_indices[i+1]
            curr_type = extrema_types[i]
            next_type = extrema_types[i+1]
            
            # Indices to fill for this segment
            idxs_to_fill = np.arange(curr_idx, next_idx + 1)
            
            # Determine Phase Range
            if curr_type == 'min' and next_type == 'max':
                # From Min to Max -> 0 to pi
                phi_start = 0
                phi_end = np.pi
            elif curr_type == 'max' and next_type == 'min':
                # From Max to Min -> pi to 2pi
                phi_start = np.pi
                phi_end = 2 * np.pi
            else:
                # Two mins or two maxes in a row (usually noise or skipped crossing)
                # Skip interpolation for this irregular segment
                continue
                
            # Linear Interpolation
            # np.linspace generates evenly spaced numbers over the interval
            phase_values = np.linspace(phi_start, phi_end, len(idxs_to_fill))
            phase_interpolated[idxs_to_fill] = phase_values
            
        # 4. Wrap Phase to [0, 2pi)
        # The Max->Min segment goes up to 2pi, which is numerically equivalent to 0.
        # Modulo ensures the values stay within standard bounds.
        final_phase = phase_interpolated % (2 * np.pi)
        
        return final_phase

    def estimate_theta_phases(self, signal: core.Signal, f_range, method = 'hilbert'):
        assert signal.n_channels == 1, "signal should have only a single trace"
        if method == 'hilbert':
            theta_phases = np.angle(signal.mne.filter(l_freq=f_range[0], h_freq=f_range[-1], verbose=False).apply_hilbert(envelope=False).get_data()[0])
            theta_phases = np.mod(theta_phases + np.pi, 2 * np.pi)
        elif method == 'wavelet':
            board_f_trace = signal.copy().mne.filter(l_freq=f_range[0], h_freq=f_range[-1], verbose=False).get_data()[0]
            theta_phases = self.get_extrema_interpolated_phase(board_f_trace)
        
        # Interpolate theta phase for each spike
        sig_t = signal.time
        phase = []
        for spiketrain in self.ratemap_spiketrains:
            phase.append(interpolate_phase_circular(sig_t, theta_phases, spiketrain))
        
        self.ratemap_spiketrains_phases = phase
    
    def plot_with_phase(
        self,
        sigma: float = 0,
        mode: str = None,
        ax=None,
        normalize: bool = True,
        stack: bool = True,
        cmap: str = "tab20b",
        subplots: tuple = None,
    ):
        cmap = mpl.cm.get_cmap(cmap)
        
        if mode is None:
            mode = self.mode
        
        # Get tuning curves and optionally smooth
        ratemaps = self.tuning_curves.copy()
        if sigma > 0:
            # Convert sigma from position units to bins
            sigma_bins = sigma / self.grid_bin
            
            if mode == 'circular':
                # Use circular boundary conditions
                ratemaps = core.Ratemap._circular_gaussian_smooth(
                    ratemaps, sigma_bins
                )
            else:
                # Use regular Gaussian smoothing
                ratemaps = gaussian_filter1d(ratemaps, sigma=sigma_bins, axis=1)
        
        # Normalize tuning curves if requested
        if normalize:
            ratemaps = np.array([
                map_ / np.max(map_) if np.max(map_) > 0 else map_
                for map_ in ratemaps
            ])
        
        # Get phase and position data
        phases = self.ratemap_spiketrains_phases
        position = self.ratemap_spiketrains_pos
        nCells = len(ratemaps)
        bin_cntr = self.coords
        
        def plot_(cell, ax, axphase):
            """Helper function to plot a single cell"""
            color = cmap(cell / nCells)
            if subplots is None:
                ax.clear()
                axphase.clear()
            
            # Plot tuning curve
            ax.fill_between(bin_cntr, 0, ratemaps[cell], color=color, alpha=0.3)
            ax.plot(bin_cntr, ratemaps[cell], color=color, alpha=0.2)
            ax.set_xlabel("Position")
            ax.set_ylabel("Normalized frate" if normalize else "frate (Hz)")
            ax.set_title(f"Cell id {self.neuron_ids[cell]}")
            if normalize:
                ax.set_ylim([0, 1])
            
            # Plot phase data
            axphase.scatter(position[cell], phases[cell], c="k", s=0.6, alpha=0.7)
            if stack:
                # Double up y-axis (convention for phase precession plots)
                axphase.scatter(position[cell], phases[cell] + 2 * np.pi, c="k", s=0.6, alpha=0.7)
            
            axphase.set_ylabel(r"$\theta$ Phase (rad)")

            # Set right y-axis tick values (optional - customize as needed)
            if stack:
                axphase.set_yticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
                axphase.set_yticklabels(['0', 'π', '2π', '3π', '4π'])
            else:
                axphase.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
                axphase.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        
        # Create figure and plot
        if ax is None:
            if subplots is None:
                # Interactive widget mode
                Fig = plotting.Fig(nrows=1, ncols=1, size=(10, 5))
                ax = plt.subplot(Fig.gs[0])
                ax.spines["right"].set_visible(True)
                axphase = ax.twinx()
                widgets.interact(
                    plot_,
                    cell=widgets.IntSlider(
                        min=0,
                        max=nCells - 1,
                        step=1,
                        description="Cell ID:",
                    ),
                    ax=widgets.fixed(ax),
                    axphase=widgets.fixed(axphase),
                )
            else:
                # Multi-panel plot mode
                ncols, nrows = subplots
                nrows = int(np.ceil(nCells / ncols))

                Fig = plotting.Fig(
                    nrows=nrows,
                    ncols=ncols,
                    size=(15, 3 * nrows)
                )

                for cell in range(nCells):
                    ax = plt.subplot(Fig.gs[cell])
                    axphase = ax.twinx()
                    plot_(cell, ax, axphase)

        else:
            # Single cell plot into provided axes
            axphase = ax.twinx()
            cell = 0
            plot_(cell, ax, axphase)
        
        return ax
    
    def analyze_phase_precession(
        self,
        pf_data: pd.DataFrame,
        height_thresh: float = 1.0,
        n_permutations: int = 1000,
        sigma: float = 0,
        plot: bool = False,
        plot_mode: str = 'interactive',
        gpu = False,
        verbose = False,
        **kwargs
    ):
        # Check if phase data exists
        if not hasattr(self, 'ratemap_spiketrains_phases'):
            raise ValueError(
                "Theta phases not computed. Call self.estimate_theta_phases() first."
            )
        
        # Filter by height threshold
        valid_mask = (pf_data['height'] >= height_thresh) & \
                    ~(pf_data['left_edge'].isna() | pf_data['right_edge'].isna())
        pf_data_filtered = pf_data[valid_mask].copy()
        
        # Sort by height (descending)
        pf_data_sorted = pf_data_filtered.sort_values('height', ascending=False).reset_index(drop=True)
        
        print(f"Analyzing phase precession for {len(pf_data_sorted)} place fields...")
        
        # Track boundaries
        track_max = 2 * np.pi if self.mode == 'circular' else np.max(self.coords)
        
        # Store results
        results_list = []
        
        # Group by cell_id to handle multiple peaks per cell
        plot_data_by_cell = {}
        
        for idx, row in tqdm(pf_data_sorted.iterrows(), total=len(pf_data_sorted)):
            cell_id = row['cell_id']
            peak_no = row['peak_no']
            height = row['height']
            center = row['center']
            left_edge = row['left_edge']
            right_edge = row['right_edge']
            
            # Find cell index
            cell_idx = np.where(self.neuron_ids == cell_id)[0]
            if len(cell_idx) == 0:
                continue
            cell_idx = cell_idx[0]
            
            # Get spike data
            spike_pos = self.ratemap_spiketrains_pos[cell_idx]
            spike_phase = self.ratemap_spiketrains_phases[cell_idx]
            
            if len(spike_pos) == 0 or len(spike_phase) == 0:
                continue
            
            # Mask spikes within place field
            if self.mode == 'circular' and right_edge < left_edge:
                # Wrap-around case
                mask = (spike_pos >= left_edge) | (spike_pos <= right_edge)
                pos_masked = spike_pos[mask]
                
                # Linearize positions
                pos_linear = np.where(
                    pos_masked >= left_edge,
                    pos_masked - left_edge,
                    pos_masked + (track_max - left_edge)
                )
            else:
                # Normal case
                mask = (spike_pos >= left_edge) & (spike_pos <= right_edge)
                pos_linear = spike_pos[mask]
            
            phase_masked = spike_phase[mask]
            
            # Need at least 3 spikes for regression
            if len(pos_linear) < 3:
                continue
            
            # Perform circular-linear regression
            try:
                if gpu:
                    fit_result = circular_linear_regression_cuda(
                        pos_linear, 
                        phase_masked,
                        n_permutations=n_permutations
                    )
                else:
                    fit_result = circular_linear_regression(
                        pos_linear, 
                        phase_masked,
                    )
                
                # Store results
                results_list.append({
                    'cell_id': cell_id,
                    'peak_no': peak_no,
                    'height': height,
                    'center': center,
                    'left_edge': left_edge,
                    'right_edge': right_edge,
                    'n_spikes': len(pos_linear),
                    'slope': fit_result['slope'],
                    'intercept': fit_result['intercept'],
                    'r_squared': fit_result['r_squared'],
                    'r_signed': fit_result['signed_r'],
                    'pvalue': fit_result['pvalue']
                })
                
                # Store plot data if needed
                if plot:
                    # Initialize cell data if first peak for this cell
                    if cell_id not in plot_data_by_cell:
                        # Get tuning curve for this cell (with smoothing if requested)
                        tuning_curve = self.tuning_curves[cell_idx].copy()
                        if sigma > 0:
                            # Convert sigma from position units to bins
                            sigma_bins = sigma / self.grid_bin
                            
                            if self.mode == 'circular':
                                tuning_curve = core.Ratemap._circular_gaussian_smooth(
                                    tuning_curve.reshape(1, -1), sigma_bins
                                ).squeeze()
                            else:
                                tuning_curve = gaussian_filter1d(tuning_curve, sigma=sigma_bins)
                        
                        # Normalize tuning curve for plotting
                        tuning_curve_normalized = tuning_curve / np.max(tuning_curve) if np.max(tuning_curve) > 0 else tuning_curve
                        
                        # Get all spikes for this cell (for scatter plot)
                        all_spike_pos = self.ratemap_spiketrains_pos[cell_idx]
                        all_spike_phase = self.ratemap_spiketrains_phases[cell_idx]
                        
                        plot_data_by_cell[cell_id] = {
                            'cell_id': cell_id,
                            'cell_idx': cell_idx,
                            'tuning_curve_normalized': tuning_curve_normalized,
                            'all_spike_pos': all_spike_pos,
                            'all_spike_phase': all_spike_phase,
                            'peaks': []
                        }
                    
                    # Add this peak's data
                    plot_data_by_cell[cell_id]['peaks'].append({
                        'peak_no': peak_no,
                        'left_edge': left_edge,
                        'right_edge': right_edge,
                        'center': center,
                        'pos': pos_linear,
                        'phase': phase_masked,
                        'fit': fit_result,
                        'n_spikes': len(pos_linear)
                    })
                    
            except Exception as e:
                print(f"Warning: Could not fit cell {cell_id}, peak {peak_no}: {e}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(results_list)
        
        if verbose == True:
            print(f"\nSuccessfully analyzed {len(results_df)} place fields")
            if len(results_df) > 0:
                n_significant = np.sum(results_df['pvalue'] < 0.05)
                print(f"Significant phase precession (p < 0.05): {n_significant} ({100*n_significant/len(results_df):.1f}%)")
                print(f"Mean slope: {np.mean(results_df['slope']):.4f} rad/unit")
                print(f"Mean R²: {np.mean(results_df['r_squared']):.3f}")
        
        # Create plots if requested
        if plot and len(plot_data_by_cell) > 0:
            subplots = kwargs.get('subplots', (4, 5))
            
            # Convert dict to list for plotting
            plot_data_list = list(plot_data_by_cell.values())
            
            if plot_mode == 'interactive':
                # Interactive widget mode
                def plot_cell(cell_idx_plot):
                    if cell_idx_plot >= len(plot_data_list):
                        return
                    
                    data = plot_data_list[cell_idx_plot]
                    n_peaks = len(data['peaks'])
                    
                    # Create figure with 1 left plot and n_peaks right plots
                    fig = plt.figure(figsize=(6 + 5*n_peaks, 5))
                    gs = GridSpec(1, 1 + n_peaks, figure=fig)
                    
                    # LEFT PLOT: Follow plot_with_phase style - tuning curve + phase scatter
                    ax_left = fig.add_subplot(gs[0, 0])
                    bin_centers = self.coords
                    cell_idx = data['cell_idx']
                    
                    # Plot tuning curve (normalized, filled)
                    color = plt.cm.tab20b(cell_idx / len(self.neuron_ids))
                    ax_left.fill_between(bin_centers, 0, data['tuning_curve_normalized'], 
                                        color=color, alpha=0.3)
                    ax_left.plot(bin_centers, data['tuning_curve_normalized'], 
                                color=color, alpha=0.2)
                    ax_left.set_xlabel("Position (cm)", fontsize=11)
                    ax_left.set_ylabel("Normalized frate", fontsize=11)
                    ax_left.set_title(f"Cell {data['cell_id']} ({n_peaks} peak{'s' if n_peaks > 1 else ''})", 
                                    fontsize=12, fontweight='bold')
                    ax_left.set_ylim([0, 1])
                    
                    # Create twin axis for phase
                    ax_phase_left = ax_left.twinx()
                    
                    # Plot ALL spikes phase scatter (stacked to 4π)
                    ax_phase_left.scatter(data['all_spike_pos'], data['all_spike_phase'], 
                                        c='k', s=0.6, alpha=0.7, zorder=3)
                    ax_phase_left.scatter(data['all_spike_pos'], data['all_spike_phase'] + 2*np.pi, 
                                        c='k', s=0.6, alpha=0.7, zorder=3)
                    
                    # Mark each peak's region with different colors
                    peak_colors = plt.cm.Set3(np.linspace(0, 1, n_peaks))
                    for peak_idx, peak_data in enumerate(data['peaks']):
                        left_edge = peak_data['left_edge']
                        right_edge = peak_data['right_edge']
                        peak_color = peak_colors[peak_idx]
                        
                        # Shade the place field region
                        if self.mode == 'circular' and right_edge < left_edge:
                            # Wrap-around: shade two regions
                            ax_left.axvspan(left_edge, bin_centers[-1], 
                                        alpha=0.3, color=peak_color, zorder=1, 
                                        label=f'Peak {peak_data["peak_no"]}')
                            ax_left.axvspan(bin_centers[0], right_edge, 
                                        alpha=0.3, color=peak_color, zorder=1)
                        else:
                            # Normal region
                            ax_left.axvspan(left_edge, right_edge, 
                                        alpha=0.3, color=peak_color, zorder=1,
                                        label=f'Peak {peak_data["peak_no"]}')
                    
                    # Format phase axis
                    ax_phase_left.set_ylim([0, 4*np.pi])
                    ax_phase_left.set_yticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
                    ax_phase_left.set_yticklabels(['0', 'π', '2π', '3π', '4π'])
                    ax_phase_left.set_ylabel(r"$\theta$ Phase (rad)", fontsize=11)
                    
                    ax_left.legend(loc='upper left', fontsize=9)
                    ax_left.grid(True, alpha=0.3)
                    
                    # RIGHT PLOTS: One for each peak
                    for peak_idx, peak_data in enumerate(data['peaks']):
                        ax_right = fig.add_subplot(gs[0, 1 + peak_idx])
                        
                        # Plot data points (stacked to 4π)
                        ax_right.scatter(peak_data['pos'], peak_data['phase'], 
                                    c='black', s=20, alpha=0.6, label='Spikes', zorder=3)
                        ax_right.scatter(peak_data['pos'], peak_data['phase'] + 2*np.pi, 
                                    c='black', s=20, alpha=0.3, zorder=3)
                        
                        # Plot fitted line
                        pos_fit = np.linspace(np.min(peak_data['pos']), np.max(peak_data['pos']), 200)
                        phase_fit = np.mod(
                            peak_data['fit']['slope'] * pos_fit + peak_data['fit']['intercept'], 
                            2*np.pi
                        )
                        
                        ax_right.scatter(pos_fit, phase_fit, c='red', s=10, alpha=0.8, 
                                    label='Fit', zorder=4, marker='.')
                        ax_right.scatter(pos_fit, phase_fit + 2*np.pi, c='red', s=10, 
                                    alpha=0.4, zorder=4, marker='.')
                        
                        # Shade the field region on phase plot
                        x_min, x_max = np.min(peak_data['pos']), np.max(peak_data['pos'])
                        peak_color = peak_colors[peak_idx]
                        ax_right.axvspan(x_min, x_max, alpha=0.2, color=peak_color, 
                                        zorder=1, label='Field region')
                        
                        # Format y-axis
                        ax_right.set_ylim([0, 4*np.pi])
                        ax_right.set_yticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
                        ax_right.set_yticklabels(['0', 'π', '2π', '3π', '4π'])
                        
                        # Labels and title
                        ax_right.set_xlabel('Position (linearized)', fontsize=11)
                        ax_right.set_ylabel('Phase (rad)', fontsize=11)
                        
                        # Title with statistics
                        slope_deg = np.degrees(peak_data['fit']['slope'])
                        title = f"Peak {peak_data['peak_no']}\n"
                        title += f"Slope: {peak_data['fit']['slope']:.4f} rad/unit\n"
                        title += f"R²: {peak_data['fit']['r_squared']:.3f} | "
                        title += f"p: {peak_data['fit']['pvalue']:.4f}\n"
                        title += f"n={peak_data['n_spikes']} spikes"
                        ax_right.set_title(title, fontsize=10)
                        
                        ax_right.legend(fontsize=9)
                        ax_right.grid(True, alpha=0.3)
                    
                    plt.show()
                
                # Create widget
                interact(
                    plot_cell,
                    cell_idx_plot=IntSlider(
                        min=0,
                        max=len(plot_data_list) - 1,
                        step=1,
                        value=0,
                        description='Cell:',
                        continuous_update=False,
                        style={'description_width': 'initial'}
                    )
                )
                
            elif plot_mode == 'grid':
                # Grid mode - each row is one cell with multiple peaks
                n_cells = len(plot_data_list)
                n_rows = subplots[0]
                
                # Calculate max peaks to determine column layout
                max_peaks = max(len(data['peaks']) for data in plot_data_list)
                
                n_figs = int(np.ceil(n_cells / n_rows))
                
                for fig_idx in range(n_figs):
                    fig = plt.figure(figsize=(5 + 4*max_peaks, 4*n_rows))
                    gs = GridSpec(n_rows, 1 + max_peaks, figure=fig, wspace=0.3, hspace=0.4)
                    
                    start_idx = fig_idx * n_rows
                    end_idx = min(start_idx + n_rows, n_cells)
                    
                    for row_idx, cell_idx_plot in enumerate(range(start_idx, end_idx)):
                        data = plot_data_list[cell_idx_plot]
                        n_peaks = len(data['peaks'])
                        
                        # LEFT: tuning curve + phase scatter
                        ax_left = fig.add_subplot(gs[row_idx, 0])
                        bin_centers = self.coords
                        cell_idx = data['cell_idx']
                        
                        # Plot tuning curve (normalized, filled)
                        color = plt.cm.tab20b(cell_idx / len(self.neuron_ids))
                        ax_left.fill_between(bin_centers, 0, data['tuning_curve_normalized'], 
                                            color=color, alpha=0.3)
                        ax_left.plot(bin_centers, data['tuning_curve_normalized'], 
                                    color=color, alpha=0.2, linewidth=1)
                        ax_left.set_ylabel("Norm. rate", fontsize=8)
                        ax_left.set_ylim([0, 1])
                        ax_left.tick_params(labelsize=7)
                        
                        # Create twin axis for phase
                        ax_phase_left = ax_left.twinx()
                        
                        # Plot ALL spikes phase scatter (stacked to 4π)
                        ax_phase_left.scatter(data['all_spike_pos'], data['all_spike_phase'], 
                                            c='k', s=0.3, alpha=0.6, zorder=3)
                        ax_phase_left.scatter(data['all_spike_pos'], data['all_spike_phase'] + 2*np.pi, 
                                            c='k', s=0.3, alpha=0.6, zorder=3)
                        
                        # Mark each peak's region with different colors
                        peak_colors = plt.cm.Set3(np.linspace(0, 1, n_peaks))
                        for peak_idx, peak_data in enumerate(data['peaks']):
                            left_edge = peak_data['left_edge']
                            right_edge = peak_data['right_edge']
                            peak_color = peak_colors[peak_idx]
                            
                            # Shade the place field region
                            if self.mode == 'circular' and right_edge < left_edge:
                                ax_left.axvspan(left_edge, bin_centers[-1], 
                                            alpha=0.3, color=peak_color, zorder=1)
                                ax_left.axvspan(bin_centers[0], right_edge, 
                                            alpha=0.3, color=peak_color, zorder=1)
                            else:
                                ax_left.axvspan(left_edge, right_edge, 
                                            alpha=0.3, color=peak_color, zorder=1)
                        
                        # Format phase axis
                        ax_phase_left.set_ylim([0, 4*np.pi])
                        ax_phase_left.set_yticks([0, 2*np.pi, 4*np.pi])
                        ax_phase_left.set_yticklabels(['0', '2π', '4π'], fontsize=7)
                        ax_phase_left.set_ylabel(r"$\theta$ Phase", fontsize=8)
                        ax_phase_left.tick_params(labelsize=7)
                        
                        ax_left.set_title(f"Cell {data['cell_id']} ({n_peaks} peak{'s' if n_peaks > 1 else ''})", 
                                        fontsize=9)
                        ax_left.set_xlabel('Position', fontsize=8)
                        ax_left.grid(True, alpha=0.3)
                        
                        # RIGHT: One subplot for each peak
                        for peak_idx, peak_data in enumerate(data['peaks']):
                            ax_right = fig.add_subplot(gs[row_idx, 1 + peak_idx])
                            
                            # Plot phase precession
                            ax_right.scatter(peak_data['pos'], peak_data['phase'], 
                                        c='black', s=5, alpha=0.6)
                            ax_right.scatter(peak_data['pos'], peak_data['phase'] + 2*np.pi, 
                                        c='black', s=5, alpha=0.3)
                            
                            pos_fit = np.linspace(np.min(peak_data['pos']), np.max(peak_data['pos']), 100)
                            phase_fit = np.mod(
                                peak_data['fit']['slope'] * pos_fit + peak_data['fit']['intercept'], 
                                2*np.pi
                            )
                            ax_right.scatter(pos_fit, phase_fit, c='red', s=3, alpha=0.8)
                            ax_right.scatter(pos_fit, phase_fit + 2*np.pi, c='red', s=3, alpha=0.4)
                            
                            ax_right.set_ylim([0, 4*np.pi])
                            ax_right.set_yticks([0, 2*np.pi, 4*np.pi])
                            ax_right.set_yticklabels(['0', '2π', '4π'], fontsize=7)
                            ax_right.set_title(f"Peak {peak_data['peak_no']}: R²={peak_data['fit']['r_squared']:.2f}, p={peak_data['fit']['pvalue']:.3f}", 
                                            fontsize=8)
                            ax_right.set_xlabel('Position', fontsize=8)
                            ax_right.set_ylabel('Phase', fontsize=8)
                            ax_right.tick_params(labelsize=7)
                            ax_right.grid(True, alpha=0.3)
                    
                    plt.show()
            
            else:
                raise ValueError(f"Unknown plot_mode: {plot_mode}. Use 'interactive' or 'grid'")
        
        # Store results in object
        self.phase_precession_results = results_df
        
        return results_df
    
    def filter_overlapping_peaks(self, pf_data, circular=True, track_length=2*np.pi, verbose=False):
        """
        Filter overlapping place field peaks using connected components.
        
        Finds groups of transitively overlapping peaks. For each group:
        1. Keeps the peak with the maximum height.
        2. Merges the edges to cover the full extent of the group (farthest left/right).
        """
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse import csr_matrix
        
        def get_circular_distance(p1, p2, L):
            diff = np.abs(p1 - p2)
            return np.minimum(diff, L - diff)

        def intervals_overlap_circular(c1, w1, c2, w2, track_length):
            """
            Check overlap using distance between centers.
            Two intervals overlap if distance(c1, c2) < (half_width1 + half_width2)
            """
            dist = get_circular_distance(c1, c2, track_length)
            # Use strictly less than for overlap (or <= if touching counts)
            return dist < (w1 + w2) / 2

        def intervals_overlap_linear(left1, right1, left2, right2):
            return not (right1 < left2 or right2 < left1)
        
        filtered_data_list = []
        
        for cell_id, cell_group in pf_data.groupby('cell_id'):
            n_peaks = len(cell_group)
            
            if n_peaks <= 1:
                filtered_data_list.append(cell_group)
                continue
            
            # Reset index 
            cell_group = cell_group.reset_index(drop=True)
            
            # Build adjacency matrix
            adjacency = np.zeros((n_peaks, n_peaks), dtype=bool)
            
            for i in range(n_peaks):
                for j in range(i + 1, n_peaks):
                    # Get Centers and Widths for distance-based check (Robust for circular)
                    c1, w1 = cell_group.loc[i, 'center'], cell_group.loc[i, 'width']
                    c2, w2 = cell_group.loc[j, 'center'], cell_group.loc[j, 'width']
                    
                    # Also keep edges for linear check
                    l1, r1 = cell_group.loc[i, 'left_edge'], cell_group.loc[i, 'right_edge']
                    l2, r2 = cell_group.loc[j, 'left_edge'], cell_group.loc[j, 'right_edge']
                    
                    if circular:
                        overlap = intervals_overlap_circular(c1, w1, c2, w2, track_length)
                    else:
                        overlap = intervals_overlap_linear(l1, r1, l2, r2)
                    
                    if overlap:
                        adjacency[i, j] = True
                        adjacency[j, i] = True
            
            # Find connected components
            n_components, labels = connected_components(csr_matrix(adjacency), directed=False)
            
            # Process each component
            peaks_to_keep_indices = []
            
            # We need to modify the dataframe, so we will collect modified rows
            modified_rows = []

            for comp_id in range(n_components):
                component_mask = labels == comp_id
                component_indices = np.where(component_mask)[0]
                
                # 1. Find the dominant peak (tallest)
                heights = cell_group.loc[component_indices, 'height'].values
                best_local_idx = component_indices[np.argmax(heights)]
                best_peak_row = cell_group.loc[best_local_idx].copy()
                
                # If there is only one peak in the group, no merging needed
                if len(component_indices) == 1:
                    modified_rows.append(best_peak_row)
                    continue

                # 2. Merge Edges (The hard part)
                # Strategy: Unwrap all intervals relative to the best peak's center
                ref_center = best_peak_row['center']
                
                unwrapped_lefts = []
                unwrapped_rights = []
                
                for idx in component_indices:
                    l = cell_group.loc[idx, 'left_edge']
                    r = cell_group.loc[idx, 'right_edge']
                    
                    if circular:
                        # Unwrap left edge relative to reference center
                        diff_l = l - ref_center
                        # Force diff into [-L/2, L/2] to handle wrap-around
                        diff_l = (diff_l + track_length/2) % track_length - track_length/2
                        uw_l = ref_center + diff_l
                        
                        # Unwrap right edge relative to reference center
                        diff_r = r - ref_center
                        diff_r = (diff_r + track_length/2) % track_length - track_length/2
                        uw_r = ref_center + diff_r
                        
                        # Correction: If the field wraps (right < left originally), 
                        # the unwrapped right must be > unwrapped left.
                        # However, calculating offsets from center usually handles this 
                        # IF the field isn't covering the whole track.
                        # Safe check: if uw_r < uw_l, add track_length to uw_r
                        if uw_r < uw_l:
                             uw_r += track_length

                        unwrapped_lefts.append(uw_l)
                        unwrapped_rights.append(uw_r)
                    else:
                        unwrapped_lefts.append(l)
                        unwrapped_rights.append(r)
                
                # Find farthest extent in unwrapped space
                min_l = min(unwrapped_lefts)
                max_r = max(unwrapped_rights)
                
                # 3. Update the best peak's edges (wrapping back if needed)
                if circular:
                    new_left = min_l % track_length
                    new_right = max_r % track_length
                    new_width = max_r - min_l # Width is linear distance in unwrapped space
                else:
                    new_left = min_l
                    new_right = max_r
                    new_width = max_r - min_l

                best_peak_row['left_edge'] = new_left
                best_peak_row['right_edge'] = new_right
                best_peak_row['width'] = new_width
                
                modified_rows.append(best_peak_row)
            
            # Reconstruct DataFrame for this cell
            filtered_data_list.append(pd.DataFrame(modified_rows))
        
        # Concatenate results
        if filtered_data_list:
            filtered_data = pd.concat(filtered_data_list, ignore_index=True)
        else:
            filtered_data = pd.DataFrame()
        
        if verbose:
            n_original = len(pf_data)
            n_filtered = len(filtered_data)
            print(f"Overlap filtering: {n_original} -> {n_filtered} peaks")
        
        return filtered_data
    
    def plot_ratemaps(self, **kwargs):
        """
        Plot tuning curves for all neurons.
        
        Wrapper for neuropy.plotting.plot_ratemap function.
        
        Args:
            **kwargs: Arguments passed to plotting.plot_ratemap
        
        Returns:
            Result from plotting.plot_ratemap
        """
        return plotting.plot_ratemap(self, **kwargs)
    
    def plot_rasters(
        self,
        jitter: float = 0,
        plot_time: bool = False,
        scale: str = None,
        sort: bool = True,
        ax=None,
    ):
        """
        Plot ratemap as a raster for each neuron.
        
        Creates spike raster plots showing spike positions for each neuron,
        optionally sorted by peak location.
        
        Args:
            jitter: Offset each neuron's spikes vertically to show density (default: 0)
            plot_time: Show timing of each spike on y-axis (default: False)
            scale: Coordinate scaling option (default: None)
                  None = keep in native coords
                  'tuning_curve' = scale to match tuning curve bins
            sort: Sort neurons by peak location (default: True)
            ax: Matplotlib axes object (default: None, creates new)
        """
        assert isinstance(ax, plt.Axes) or (ax is None)
        assert (scale == "tuning_curve") or (scale is None)
        
        if ax is None:
            _, ax = plt.subplots()
        
        # Get neuron order (sorted by peak or original)
        order = self.get_sort_order(by="index") if sort else np.arange(self.n_neurons)
        spiketrains_pos = [self.ratemap_spiketrains_pos[i] for i in order]
        spiketrains_t = [self.ratemap_spiketrains[i] for i in order]
        
        # Calculate scale factor if needed
        scale_factor = 1
        if scale == "tuning_curve":
            ncm = np.ptp(self.coords)
            nbins = self.tuning_curves.shape[1]
            scale_factor = (nbins - 0) / ncm
        
        # Plot each neuron's spikes
        for i, (spk_pos, spk_t) in enumerate(zip(spiketrains_pos, spiketrains_t)):
            if plot_time:
                # Y-position represents spike time
                ypos = (spk_t - self.t_start) / ((self.t_stop - self.t_start) * 1.1) + i - 0.45
                ypos_traj = (self.t - self.t_start) / ((self.t_stop - self.t_start) * 1.1) + i - 0.45
                ax.plot(self.x * scale_factor - 0.5, ypos_traj, "-", color=[0, 0, 1, 0.3])
            else:
                # Y-position is neuron index with optional jitter
                ypos = i * np.ones_like(spk_pos) + np.random.randn(spk_pos.shape[0]) * jitter
            
            ax.plot(spk_pos * scale_factor, ypos, "k.", markersize=1)
    
    def plot_ratemap_w_raster(self, ind: int = None, id: int = None, ax=None, **kwargs):
        """
        Plot tuning curve with spike raster below for a single neuron.
        
        Creates a two-panel plot showing the firing rate tuning curve above
        and spike raster below.
        
        Args:
            ind: Neuron index (default: None)
            id: Neuron ID (default: None)
                NOTE: Exactly one of 'ind' or 'id' must be provided
            ax: List of 2 matplotlib axes (default: None, creates new)
            **kwargs: Additional arguments (currently unused)
        
        Returns:
            List of matplotlib axes objects
        """
        # Validate input
        assert (ind is None) != (id is None), "Exactly one of 'ind' and 'id' must be provided"
        
        # Convert ID to index if needed
        if ind is None:
            ind = np.where(id == self.neuron_ids)[0][0]
        
        # Slice desired neuron's placefield
        pfuse = self.neuron_slice([ind])
        
        # Create axes if not provided
        if ax is None:
            _, ax = plt.subplots(
                2, 1, figsize=(4, 4), sharex=True, height_ratios=[3, 2]
            )
        
        # Plot tuning curve on top
        pfuse.plot_ratemaps(normalize_tuning_curve=True, ax=ax[0])
        
        # Plot raster below
        pfuse.plot_rasters(plot_time=True, ax=ax[1])
        
        return ax
    
    def plot_raw_ratemaps_laps(self, ax=None, subplots: tuple = (8, 9)):
        """
        Plot raw ratemaps by laps.
        
        Wrapper for plotting.plot_raw_ratemaps function.
        
        Args:
            ax: Matplotlib axes (default: None)
            subplots: Grid layout (rows, cols) (default: (8, 9))
        
        Returns:
            Result from plotting.plot_raw_ratemaps
        """
        return plotting.plot_raw_ratemaps()
    
    def neuron_slice(self, inds: list = None, ids: list = None):
        """
        Create a new Pf1D object with a subset of neurons.
        
        Returns a deep copy of the current object containing only the specified
        neurons. Useful for analyzing or plotting specific cells.
        
        Args:
            inds: List of neuron indices (default: None)
            ids: List of neuron IDs (default: None)
                NOTE: Exactly one of 'inds' or 'ids' must be provided
        
        Returns:
            New Pf1D object containing only the selected neurons
        """
        assert (inds is None) != (ids is None), "Exactly one of 'inds' and 'ids' must be a list or array"
        
        # Convert IDs to indices if needed
        if ids is not None:
            inds = [
                np.where(idd == self.neuron_ids)[0][0]
                for idd in np.atleast_1d(ids)
            ]
        inds = np.sort(np.atleast_1d(inds))
        
        # Make a deep copy and slice data
        pfslice = deepcopy(self)
        pfslice.tuning_curves = self.tuning_curves[inds]
        pfslice.neuron_ids = self.neuron_ids[inds]
        pfslice.ratemap_spiketrains = [self.ratemap_spiketrains[ind] for ind in inds]
        pfslice.ratemap_spiketrains_pos = [self.ratemap_spiketrains_pos[ind] for ind in inds]
        
        return pfslice
    
    def get_pf_data(self, sigma: float = 0.1, plot: bool = False, plot_mode: str = 'interactive', **kwargs):
        """
        Get comprehensive place field statistics for all neurons.
        
        Extracts peak heights, prominences, centers, edges, and widths for all
        place fields across all neurons.
        
        Args:
            sigma: Smoothing kernel width in bins (default: 1.5)
            plot: Plot all place fields with peaks and widths overlaid (default: False)
            plot_mode: How to display plots when plot=True (default: 'grid')
                    Options:
                    - 'grid': Multi-panel grid layout (rows x cols)
                    - 'interactive': Interactive widget to browse cells
            **kwargs: Additional arguments for get_pf_peaks, get_pf_widths, or plotting
                    Valid keys: 
                    - For get_pf_peaks: 'prominence', 'height', 'distance', 'width', 
                    'wlen', 'rel_height', 'plateau_size'
                    - For get_pf_widths: 'height_thresh'
                    - For plotting: 'subplots' (tuple for grid layout, default (5, 8))
        
        Returns:
            DataFrame with columns: cell_id, peak_no, height, prominence, 
                                center (in position units), width (in position units), 
                                left_edge (in position units), right_edge (in position units)
        """
        # Parse kwargs for different functions
        peaks_keys = ["prominence", "height", "distance", "width", "wlen", 
                    "rel_height", "plateau_size"]
        kwargs_peaks = {key: value for key, value in kwargs.items() if key in peaks_keys}
        kwargs_widths = {key: value for key, value in kwargs.items() if key in ["height_thresh"]}
        
        # Get plotting parameters
        subplots = kwargs.get('subplots', (5, 8))
        height_thresh = kwargs_widths.get('height_thresh', 0.5)
        
        # Get bin centers in position units
        bin_centers = self.coords
        
        # Store data for plotting
        plot_data = []
        
        # Loop through each neuron and calculate peak/width information
        pf_stats_list = []
        ind = 0
        for nid in tqdm(self.neuron_ids):
            # Get peak information
            heights, prominences, centers, tuning_curve = self.get_pf_peaks(
                cell_id=nid, sigma=sigma, **kwargs_peaks
            )
            
            # Get width information
            widths, edges = self.get_pf_widths(
                tuning_curve.squeeze(), heights, prominences, centers,
                **kwargs_widths
            )
            
            # Store plot data if plotting is requested
            if plot:
                plot_data.append({
                    'cell_id': nid,
                    'tuning_curve': tuning_curve,
                    'heights': heights,
                    'prominences': prominences,
                    'centers': centers,
                    'widths': widths,
                    'edges': edges
                })
            
            # Store results for each peak
            for idp, (height, prom, cent, width, edge) in enumerate(
                zip(heights, prominences, centers, widths, edges)
            ):
                # Convert from bin indices to position units using bin_centers
                center_pos = np.interp(cent, np.arange(len(bin_centers)), bin_centers)
                width_pos = width * self.grid_bin
                left_edge_pos = np.interp(edge[0], np.arange(len(bin_centers)), bin_centers) if not np.isnan(edge[0]) else np.nan
                right_edge_pos = np.interp(edge[1], np.arange(len(bin_centers)), bin_centers) if not np.isnan(edge[1]) else np.nan
                
                pf_stats_list.append(
                    pd.DataFrame({
                        "cell_id": nid,
                        "peak_no": idp,
                        "height": height,
                        "prominence": prom,
                        "center": center_pos,
                        "width": width_pos,
                        "left_edge": left_edge_pos,
                        "right_edge": right_edge_pos
                    }, index=[ind])
                )
                ind += 1
        
        # Create plots if requested
        if plot and len(plot_data) > 0:
            if plot_mode == 'grid':
                # Multi-panel grid layout
                n_cells = len(plot_data)
                n_rows, n_cols = subplots
                n_per_fig = n_rows * n_cols
                n_figs = int(np.ceil(n_cells / n_per_fig))
                
                for fig_idx in range(n_figs):
                    fig = plt.figure(figsize=(15, 10))
                    gs = GridSpec(n_rows, n_cols, figure=fig)
                    
                    start_idx = fig_idx * n_per_fig
                    end_idx = min(start_idx + n_per_fig, n_cells)
                    
                    for subplot_idx, cell_idx in enumerate(range(start_idx, end_idx)):
                        ax = fig.add_subplot(gs[subplot_idx])
                        data = plot_data[cell_idx]
                        
                        self.plot_pf_peaks_and_widths(
                            data['tuning_curve'],
                            data['widths'],
                            data['edges'],
                            data['heights'],
                            data['prominences'],
                            data['centers'],
                            height_thresh,
                            ax=ax,
                            title=f"Cell {data['cell_id']}"
                        )
                    
                    plt.show()
                    
            elif plot_mode == 'interactive':
                # Interactive widget
                
                n_cells = len(plot_data)
                fig, ax = plt.subplots(figsize=(12, 6))
                
                def plot_cell(cell_idx):
                    ax.clear()
                    data = plot_data[cell_idx]
                    
                    self.plot_pf_peaks_and_widths(
                        data['tuning_curve'],
                        data['widths'],
                        data['edges'],
                        data['heights'],
                        data['prominences'],
                        data['centers'],
                        height_thresh,
                        ax=ax,
                        title=f"Cell {data['cell_id']}"
                    )
                    fig.canvas.draw()
                
                interact(
                    plot_cell,
                    cell_idx=IntSlider(
                        min=0,
                        max=n_cells - 1,
                        step=1,
                        value=0,
                        description='Cell:',
                        continuous_update=False
                    )
                )
            else:
                raise ValueError(f"Unknown plot_mode: {plot_mode}. Use 'grid' or 'interactive'")
        
        # Combine results into DataFrame
        df_result = pd.concat(pf_stats_list, axis=0) if pf_stats_list else pd.DataFrame()
        
        # --- NEW CODE START ---
        # Exclude rows where width or edges are NaN
        if not df_result.empty:
            df_result = df_result.dropna(subset=['width', 'left_edge', 'right_edge'])
            # Reset index to fix the gaps created by dropping rows
            df_result = df_result.reset_index(drop=True)
        # --- NEW CODE END ---
        
        return df_result
    
    def get_pf_peaks(
        self,
        cell_ind: int = None,
        cell_id: int = None,
        sigma: float = 1.5,
        prominence: float = 0,
        height: float = None,
        distance: int = None,
        **kwargs
    ):
        """
        Detect place field peaks using scipy.signal.find_peaks.
        
        Applies the same smoothing method (linear or circular) as specified during
        initialization, then detects peaks using scipy's find_peaks algorithm.
        
        For circular mode, extends the tuning curve [curve, curve, curve] to detect
        wrap-around fields, then only keeps peaks in the middle copy.
        
        Args:
            cell_ind: Cell index to analyze (default: None)
            cell_id: Cell ID to analyze (default: None)
                    NOTE: Exactly one of cell_ind or cell_id must be provided
            sigma: Gaussian smoothing kernel size in position units (default: 1.5)
                Will be divided by grid_bin to convert to bins
            prominence: Minimum prominence required for peaks (default: 0)
            height: Minimum height required for peaks (default: None)
            distance: Minimum distance between peaks in bins (default: None)
            **kwargs: Additional arguments passed to scipy.signal.find_peaks
        
        Returns:
            Tuple of (heights, prominences, centers, tuning_curve):
                - heights: np.ndarray of peak heights
                - prominences: np.ndarray of peak prominences
                - centers: np.ndarray of peak center positions (in bins, in [0, n_bins) range)
                - tuning_curve: Smoothed tuning curve used for analysis (original, not extended)
        """
        # Get single neuron data
        pf_use = self.neuron_slice(inds=cell_ind, ids=cell_id)
        tuning_curve = pf_use.tuning_curves.squeeze()
        
        # Apply smoothing if requested, following initialization settings
        if sigma > 0:
            # Convert sigma from position units to bins
            sigma_bins = sigma / self.grid_bin
            
            if self.mode == 'circular':
                # Use circular boundary conditions
                tuning_curve = core.Ratemap._circular_gaussian_smooth(
                    tuning_curve.reshape(1, -1), sigma_bins
                ).squeeze()
            else:
                # Use regular Gaussian smoothing
                tuning_curve = gaussian_filter1d(tuning_curve, sigma=sigma_bins)
        
        n_bins = len(tuning_curve)
        
        if self.mode == 'circular':
            # Extend tuning curve for wrap-around peak detection
            # [curve, curve, curve] allows detection of peaks spanning boundaries
            tuning_curve_help = tuning_curve.copy()
            tuning_curve_help[np.argmax(tuning_curve_help)] += 1
            tuning_curve_extended = np.concatenate([tuning_curve_help, tuning_curve, tuning_curve_help])
            
            # Find peaks in extended curve
            peak_indices_extended, properties = find_peaks(
                tuning_curve_extended,
                prominence=prominence,
                height=height,
                distance=distance,
                **kwargs
            )
            
            # Only keep peaks in the middle copy [n_bins, 2*n_bins)
            middle_mask = (peak_indices_extended >= n_bins) & (peak_indices_extended < 2 * n_bins)
            peak_indices_middle = peak_indices_extended[middle_mask]
            
            # Convert back to original [0, n_bins) range
            peak_indices = peak_indices_middle - n_bins
            
            # Extract properties for middle peaks only
            centers = peak_indices
            heights = tuning_curve[peak_indices]
            prominences = properties['prominences'][middle_mask]
            
        else:
            # Linear mode - find peaks directly
            peak_indices, properties = find_peaks(
                tuning_curve,
                prominence=prominence,
                height=height,
                distance=distance,
                **kwargs
            )
            
            # Extract peak properties
            centers = peak_indices
            heights = tuning_curve[peak_indices]
            prominences = properties['prominences']
        
        return heights, prominences, centers, tuning_curve

    def get_pf_widths(
        self,
        tuning_curve: np.ndarray,
        heights: np.ndarray,
        prominences: np.ndarray,
        centers: np.ndarray,
        height_thresh: float = 0.5,
        mode: str = None,
    ):
        """
        Calculate place field widths from peak data.
        
        Determines the width of each place field at a specified height threshold
        relative to the peak. Width is calculated by finding where the tuning curve
        crosses the threshold height on both sides of the peak.
        
        Supports circular boundary conditions where fields can wrap around the edges.
        
        Args:
            tuning_curve: Smoothed 1D tuning curve from get_pf_peaks
            heights: Peak heights from get_pf_peaks
            prominences: Peak prominences from get_pf_peaks
            centers: Peak center positions from get_pf_peaks (in bins)
            height_thresh: Relative height threshold for width calculation (0-1)
                        1.0 = at peak, 0.0 = at base (default: 0.5)
            circular: Use circular boundary conditions (default: None, uses self.circular)
        
        Returns:
            Tuple of (widths, edges):
                - widths: np.ndarray of field widths in bins
                - edges: np.ndarray (Nx2) of [left_edge, right_edge] positions
                        For circular mode, edges are in [0, n_bins) range
                        np.nan indicates edge extends beyond data limits (linear mode only)
        """
        assert tuning_curve.ndim == 1, "Tuning curve must be 1-dimensional"
        
        if mode is None:
            mode = self.mode
        
        n_bins = len(tuning_curve)
        edges, widths = [], []
        
        if mode == 'circular':
            # Extend tuning curve for wrap-around detection
            tuning_curve_extended = np.concatenate([tuning_curve, tuning_curve, tuning_curve])
            
            # Adjust centers to middle copy
            centers_extended = centers + n_bins
            track_width = 3 * n_bins
            
            # Calculate width for each peak
            for height, pro, center_orig, center_ext in zip(heights, prominences, centers, centers_extended):
                # Calculate threshold level
                thresh_level = height - pro * (1 - height_thresh)
                
                # Identify regions above threshold in extended curve
                abv_thresh_regions = contiguous_regions(tuning_curve_extended - thresh_level > 0)
                
                # Find region containing the peak center (in middle copy)
                try:
                    region_mask = np.array([
                        (center_ext >= lims[0]) & (center_ext <= lims[1])
                        for lims in abv_thresh_regions
                    ])
                    
                    if not np.any(region_mask):
                        raise ValueError("No region contains peak")
                    
                    left_ind, right_ind = abv_thresh_regions[region_mask].squeeze()
                    
                    # Interpolate exact crossing points
                    left_edge, right_edge = np.nan, np.nan
                    
                    # Left edge interpolation
                    if left_ind > 0:
                        left_edge = np.interp(
                            0,
                            tuning_curve_extended[[left_ind - 1, left_ind]] - thresh_level,
                            [left_ind - 1, left_ind]
                        )
                    else:
                        left_edge = 0
                    
                    # Right edge interpolation
                    if right_ind < track_width:
                        right_edge = np.interp(
                            0,
                            tuning_curve_extended[[right_ind, right_ind - 1]] - thresh_level,
                            [right_ind, right_ind - 1]
                        )
                    else:
                        right_edge = track_width
                    
                    # Only keep results where peak is in middle copy
                    if left_edge >= n_bins and right_edge <= 2 * n_bins:
                        # Peak and both edges in middle copy - standard case
                        left_edge_orig = left_edge - n_bins
                        right_edge_orig = right_edge - n_bins
                        width_use = right_edge_orig - left_edge_orig
                        
                    elif left_edge < n_bins and right_edge <= 2 * n_bins:
                        # Left edge in first copy, right edge in middle - wrap-around field
                        left_edge_orig = left_edge
                        right_edge_orig = right_edge - n_bins
                        width_use = right_edge - left_edge
                        
                    elif left_edge >= n_bins and right_edge > 2 * n_bins:
                        # Left edge in middle, right edge in third copy - wrap-around field
                        left_edge_orig = left_edge - n_bins
                        right_edge_orig = right_edge - 2 * n_bins
                        width_use = right_edge - left_edge
                        
                    else:
                        # Edge case: field spans more than expected
                        left_edge_orig = np.nan
                        right_edge_orig = np.nan
                        width_use = np.nan
                    
                    # Convert edges back to [0, n_bins) range using modular arithmetic
                    if not np.isnan(left_edge_orig):
                        left_edge_orig = np.mod(left_edge_orig, n_bins)
                    if not np.isnan(right_edge_orig):
                        right_edge_orig = np.mod(right_edge_orig, n_bins)
                        
                except (ValueError, IndexError):
                    # Width is less than one bin at threshold or other error
                    left_edge_orig, right_edge_orig, width_use = np.nan, np.nan, np.nan
                
                widths.append(width_use)
                edges.append(np.array([left_edge_orig, right_edge_orig]))
        
        else:
            # Linear mode (original implementation)
            track_width = n_bins
            
            # Calculate width for each peak
            for height, pro, center in zip(heights, prominences, centers):
                # Calculate threshold level
                thresh_level = height - pro * (1 - height_thresh)
                
                # Identify regions above threshold
                abv_thresh_regions = contiguous_regions(tuning_curve - thresh_level > 0)
                
                # Find region containing the peak center
                try:
                    left_ind, right_ind = abv_thresh_regions[
                        np.array([
                            (center > lims[0]) & (center < lims[1])
                            for lims in abv_thresh_regions
                        ])
                    ].squeeze()
                    
                    # Interpolate exact crossing points
                    left_edge, right_edge = np.nan, np.nan
                    
                    # Left edge interpolation
                    if left_ind > 0:
                        left_edge = np.interp(
                            0,
                            tuning_curve[[left_ind - 1, left_ind]] - thresh_level,
                            [left_ind - 1, left_ind]
                        )
                    
                    # Right edge interpolation
                    if right_ind < track_width:
                        right_edge = np.interp(
                            0,
                            tuning_curve[[right_ind, right_ind - 1]] - thresh_level,
                            [right_ind, right_ind - 1]
                        )
                    
                    # Calculate width handling edge cases
                    if np.isnan(left_edge):
                        width_use = right_edge
                    elif np.isnan(right_edge):
                        width_use = track_width - left_edge
                    else:
                        width_use = right_edge - left_edge
                        
                except ValueError:
                    # Width is less than one bin at threshold - set to nan
                    left_edge, right_edge, width_use = np.nan, np.nan, np.nan
                
                widths.append(width_use)
                edges.append(np.array([left_edge, right_edge]))
        
        return np.array(widths), np.array(edges)
    
    def plot_pf_peaks_and_widths(
        self,
        tuning_curve: np.ndarray,
        widths: list,
        edges: list,
        heights: np.ndarray,
        prominences: np.ndarray,
        centers: np.ndarray,
        height_thresh: float,
        mode: str = None,
        track_width: int = None,
        ax=None,
        title=None
    ):
        """
        Visualize place field peaks and widths.
        
        Plot tuning curve with identified peaks, prominences,
        and widths overlaid for quality control and visualization.
        
        Args:
            tuning_curve: 1D tuning curve
            widths: List of field widths
            edges: List of [left, right] edge positions (in bins)
            heights: Peak heights
            prominences: Peak prominences
            centers: Peak center positions (in bins)
            height_thresh: Height threshold used for width calculation (0-1)
            circular: Use circular boundary conditions (default: None, uses self.circular)
            track_width: Total track width in bins (default: None, infer from tuning_curve)
            ax: Matplotlib axes (default: None, creates new)
            title: Plot title (default: None)
        """
        # Create axes if needed
        if ax is None:
            _, ax = plt.subplots()
        
        if mode is None:
            mode = self.mode
        
        if track_width is None:
            track_width = tuning_curve.size
        
        # Get position coordinates
        x_coords = self.coords
        
        # Plot tuning curve
        ax.plot(x_coords, tuning_curve, ".-")
        
        # Plot each peak with prominence and width
        for width, edge, height, pro, center in zip(
            widths, edges, heights, prominences, centers
        ):
            # Convert center from bin index to position
            center_pos = x_coords[int(center)]
            
            # Plot prominence as vertical line
            ax.plot([center_pos, center_pos], [height - pro, height], 'k:')
            
            # Plot width as horizontal line
            if ~np.isnan(width):
                left_edge_bin = edge[0]
                right_edge_bin = edge[1]
                
                # Handle edge cases and circular wrap-around
                if mode:
                    # Check if field wraps around boundary
                    if not np.isnan(left_edge_bin) and not np.isnan(right_edge_bin):
                        # Convert bin indices to positions using interpolation
                        left_edge_pos = np.interp(left_edge_bin, np.arange(len(x_coords)), x_coords)
                        right_edge_pos = np.interp(right_edge_bin, np.arange(len(x_coords)), x_coords)
                        
                        width_height = height - pro * (1 - height_thresh)
                        
                        # Check if wraps around (right < left in circular coordinates)
                        if right_edge_bin < left_edge_bin:
                            # Field wraps around - plot in two segments
                            # Segment 1: from left edge to end
                            ax.plot([left_edge_pos, x_coords[-1]], 
                                [width_height, width_height], 'r')
                            # Segment 2: from start to right edge
                            ax.plot([x_coords[0], right_edge_pos], 
                                [width_height, width_height], 'r')
                        else:
                            # Normal case - single segment
                            ax.plot([left_edge_pos, right_edge_pos],
                                [width_height, width_height], 'r')
                else:
                    # Linear mode
                    if np.isnan(left_edge_bin):
                        left_edge_pos = x_coords[0]
                    else:
                        left_edge_pos = np.interp(left_edge_bin, np.arange(len(x_coords)), x_coords)
                    
                    if np.isnan(right_edge_bin):
                        right_edge_pos = x_coords[-1]
                    else:
                        right_edge_pos = np.interp(right_edge_bin, np.arange(len(x_coords)), x_coords)
                    
                    width_height = height - pro * (1 - height_thresh)
                    ax.plot([left_edge_pos, right_edge_pos],
                        [width_height, width_height], 'r')
        
        # Add labels
        ax.set_xlabel('Position')
        ax.set_ylabel('Firing Rate (Hz)')
        
        # Add title if provided
        if title is not None:
            ax.set_title(title)
        
        ax.grid(True, alpha=0.3)

    def peak_locations(self, by="index", n_interp=1000, **kwargs):
        """
        Wrapper for core.Ratemap.peak_locations.
        Automatically injects self.mode into the parent method.
        """
        # Call the parent method, passing self.mode explicitly
        return super().peak_locations(by=by, mode=self.mode, n_interp=n_interp)

    def get_sort_order(self, by="index", **kwargs):
        """
        Wrapper for core.Ratemap.get_sort_order.
        Automatically injects self.mode into the parent method.
        """
        return super().get_sort_order(by=by, mode=self.mode)

class Pf2D:
    def __init__(
        self,
        neurons: core.Neurons,
        position: core.Position,
        epochs: core.Epoch = None,
        frate_thresh=1.0,
        speed_thresh=3,
        grid_bin=1,
        sigma=1,
    ):
        """Calculates 2D placefields
        Parameters
        ----------
        period : list/array
            in seconds, time period between which placefields are calculated
        gridbin : int, optional
            bin size of grid in centimeters, by default 10
        speed_thresh : int, optional
            speed threshold in cm/s, by default 10 cm/s
        Returns
        -------
        [type]
            [description]
        """
        assert position.ndim > 1, "Position is not 2D"
        period = [position.t_start, position.t_stop]
        #smooth_ = lambda f: gaussian_filter1d(
        #    f, sigma / grid_bin, axis=-1
        #)  # divide by grid_bin to account for discrete spacing
        smooth_ = lambda f: gaussian_filter(f, sigma=(sigma / grid_bin, 
                                                      sigma / grid_bin))

        spikes = neurons.time_slice(*period).spiketrains
        cell_ids = neurons.neuron_ids
        nCells = len(spikes)

        # ----- Position---------
        xcoord = position.x
        ycoord = position.z #default for optitrack input
        time = position.time
        trackingRate = position.sampling_rate

        ind_maze = np.where((time > period[0]) & (time < period[1]))
        x = xcoord[ind_maze]
        y = ycoord[ind_maze]
        t = time[ind_maze]

        x_grid = np.arange(np.nanmin(x), np.nanmax(x) + grid_bin, grid_bin)
        y_grid = np.arange(np.nanmin(y), np.nanmax(y) + grid_bin, grid_bin)
        # x_, y_ = np.meshgrid(x_grid, y_grid)

        diff_posx = np.diff(x)
        diff_posy = np.diff(y)

        speed = np.sqrt(diff_posx**2 + diff_posy**2) / (1 / trackingRate)

        #speed = smooth_(speed)

        dt = t[1] - t[0]
        running = np.where(speed / dt > speed_thresh)[0]

        x_thresh = x[running]
        y_thresh = y[running]
        t_thresh = t[running]

        def make_pfs(
            t_, x_, y_, spkAll_, occupancy_, speed_thresh_, maze_, x_grid_, y_grid_
        ):
            maps, spk_pos, spk_t = [], [], []
            for cell in spkAll_:
                # assemble spikes and position data
                spk_maze = cell[np.where((cell > maze_[0]) & (cell < maze_[1]))]
                spk_speed = np.interp(spk_maze, t_[1:], speed)
                spk_y = np.interp(spk_maze, t_, y_)
                spk_x = np.interp(spk_maze, t_, x_)

                # speed threshold
                spd_ind = np.where(spk_speed > speed_thresh_)
                # spk_spd = spk_speed[spd_ind]
                spk_x = spk_x[spd_ind]
                spk_y = spk_y[spd_ind]

                # Calculate maps
                spk_map = np.histogram2d(spk_x, spk_y, bins=(x_grid_, y_grid_))[0]
                spk_map = smooth_(spk_map)
                maps.append(spk_map / occupancy_)

                spk_t.append(spk_maze[spd_ind])
                spk_pos.append([spk_x, spk_y])

            return maps, spk_pos, spk_t

        # --- occupancy map calculation -----------
        # NRK todo: might need to normalize occupancy so sum adds up to 1
        occupancy = np.histogram2d(x_thresh, y_thresh, bins=(x_grid, y_grid))[0]
        occupancy = occupancy / trackingRate + 10e-16  # converting to seconds
        occupancy = smooth_(occupancy)

        maps, spk_pos, spk_t = make_pfs(
            t, x, y, spikes, occupancy, speed_thresh, period, x_grid, y_grid
        )

        # ---- cells with peak frate abouve thresh ------
        good_cells_indx = [
            cell_indx
            for cell_indx in range(nCells)
            if np.max(maps[cell_indx]) > frate_thresh
        ]

        get_elem = lambda list_: [list_[_] for _ in good_cells_indx]

        self.spk_pos = get_elem(spk_pos)
        self.spk_t = get_elem(spk_t)
        self.ratemaps = get_elem(maps)
        self.cell_ids = cell_ids[good_cells_indx]
        self.occupancy = occupancy
        self.speed = speed
        self.x = x
        self.y = y
        self.t = t
        self.xgrid = x_grid
        self.ygrid = y_grid
        self.gridbin = grid_bin
        self.speed_thresh = speed_thresh
        self.period = period
        self.frate_thresh = frate_thresh
        self.mesh = np.meshgrid(
            self.xgrid[:-1] + self.gridbin / 2,
            self.ygrid[:-1] + self.gridbin / 2,
        )
        ngrid_centers_x = self.mesh[0].size
        ngrid_centers_y = self.mesh[1].size
        x_center = np.reshape(self.mesh[0], [ngrid_centers_x, 1], order="F")
        y_center = np.reshape(self.mesh[1], [ngrid_centers_y, 1], order="F")
        xy_center = np.hstack((x_center, y_center))
        self.gridcenter = xy_center.T

    def plotMap(self, subplots=(7, 4), fignum=None):
        """Plots heatmaps of placefields with peak firing rate
        Parameters
        ----------
        speed_thresh : bool, optional
            [description], by default False
        subplots : tuple, optional
            number of cells within each figure window. If cells exceed the number of subplots, then cells are plotted in successive figure windows of same size, by default (10, 8)
        fignum : int, optional
            figure number to start from, by default None
        """

        map_use, thresh = self.ratemaps, self.speed_thresh

        nCells = len(map_use)
        nfigures = nCells // np.prod(subplots) + 1

        if fignum is None:
            if f := plt.get_fignums():
                fignum = f[-1] + 1
            else:
                fignum = 1

        figures, gs = [], []
        for fig_ind in range(nfigures):
            fig = plt.figure(fignum + fig_ind, figsize=(6, 10), clear=True)
            gs.append(GridSpec(subplots[0], subplots[1], figure=fig))
            fig.subplots_adjust(hspace=0.4)
            fig.suptitle(
                "Place maps with peak firing rate (speed_threshold = "
                + str(thresh)
                + ")"
            )
            figures.append(fig)

        for cell, pfmap in enumerate(map_use):

            ind = cell // np.prod(subplots)
            subplot_ind = cell % np.prod(subplots)
            ax1 = figures[ind].add_subplot(gs[ind][subplot_ind])
            im = ax1.pcolorfast(
                self.xgrid,
                self.ygrid,
                np.rot90(np.fliplr(pfmap)) / np.max(pfmap),
                cmap="jet",
                vmin=0,
            )  # rot90(flipud... is necessary to match plotRaw configuration.
            # max_frate =
            ax1.axis("off")
            ax1.set_title(
                f"Cell {self.cell_ids[cell]} \n{round(np.nanmax(pfmap),2)} Hz"
            )

            # cbar_ax = fig.add_axes([0.9, 0.3, 0.01, 0.3])
            # cbar = fig.colorbar(im, cax=cbar_ax)
            # cbar.set_label("firing rate (Hz)")

    def plotRaw(
        self,
        subplots=(10, 8),
        fignum=None,
        alpha=0.5,
        label_cells=False,
        ax=None,
        clus_use=None,
    ):
        if ax is None:
            fig = plt.figure(fignum, figsize=(6, 10))
            gs = GridSpec(subplots[0], subplots[1], figure=fig)
            # fig.subplots_adjust(hspace=0.4)
        else:
            assert len(ax) == len(
                clus_use
            ), "Number of axes must match number of clusters to plot"
            fig = ax[0].get_figure()

        spk_pos_use = self.spk_pos

        if clus_use is not None:
            spk_pos_tmp = spk_pos_use
            spk_pos_use = []
            [spk_pos_use.append(spk_pos_tmp[a]) for a in clus_use]

        for cell, (spk_x, spk_y) in enumerate(spk_pos_use):
            if ax is None:
                ax1 = fig.add_subplot(gs[cell])
            else:
                ax1 = ax[cell]
            ax1.plot(self.x, self.y, color="#d3c5c5")
            ax1.plot(spk_x, spk_y, ".r", markersize=0.8)  #, color=[1, 0, 0, alpha])
            ax1.axis("off")
            if label_cells:
                # Put info on title
                info = self.cell_ids[cell]
                ax1.set_title(f"Cell {info}")

        fig.suptitle(
            f"Place maps for cells with their peak firing rate (frate thresh={self.frate_thresh},speed_thresh={self.speed_thresh})"
        )

    def plotRaw_v_time(self, cellind, speed_thresh=False, alpha=0.5, ax=None):
        if ax is None:
            fig, ax = plt.subplots(2, 1, sharex=True)
            fig.set_size_inches([10,7]) #([23, 9.7])

        # plot trajectories
        for a, pos, ylabel in zip(
            ax, [self.x, self.y], ["X position (cm)", "Y position (cm)"]
        ):
            a.plot(self.t, pos)
            a.set_xlabel("Time (seconds)")
            a.set_ylabel(ylabel)
            # pretty_plot(a)

        # Grab correct spike times/positions
        if speed_thresh:
            spk_pos_, spk_t_ = self.run_spk_pos, self.run_spk_t
        else:
            spk_pos_, spk_t_ = self.spk_pos, self.spk_t

        # plot spikes on trajectory
        for a, pos in zip(ax, spk_pos_[cellind]):
            a.plot(spk_t_[cellind], pos, "r.", color=[1, 0, 0, alpha])

        # Put info on title
        ipbool = self._obj.spikes.pyrid[cellind] == self._obj.spikes.info.index
        info = self._obj.spikes.info.iloc[ipbool]
        ax[0].set_title(
            "Cell "
            + str(info["id"])
            + ": q = "
            + str(info["q"])
            + ", speed_thresh="
            + str(self.speed_thresh)
        )

    def plot_all(self, cellind, speed_thresh=True, alpha=0.4, fig=None):
        if fig is None:
            fig_use = plt.figure(figsize=[28.25, 11.75])
        else:
            fig_use = fig
        gs = GridSpec(2, 4, figure=fig_use)
        ax2d = fig_use.add_subplot(gs[0, 0])
        axccg = np.asarray(fig_use.add_subplot(gs[1, 0]))
        axx = fig_use.add_subplot(gs[0, 1:])
        axy = fig_use.add_subplot(gs[1, 1:], sharex=axx)

        self.plotRaw(speed_thresh=speed_thresh, clus_use=[cellind], ax=[ax2d])
        self.plotRaw_v_time(
            cellind, speed_thresh=speed_thresh, ax=[axx, axy], alpha=alpha
        )
        self._obj.spikes.plot_ccg(clus_use=[cellind], type="acg", ax=axccg)

        return fig_use

