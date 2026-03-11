import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from tqdm import tqdm
import torch
import torch.nn.functional as F
import scipy.signal as sg
from typing import Union, Tuple, Optional
from .. import core
from .. import plotting
from neuropy.utils.decoder import radon_transform, radon_transform_batch, wcorr, wcorr_batch, jump_distance, column_shift

opt_radon_transform = torch.compile(radon_transform_batch, mode="default")
opt_wcorr = torch.compile(wcorr_batch, mode="default")

def crop_posterior(posterior, window_fraction=0.33):
    """
    Robust crop_posterior that handles both PyTorch Tensors (GPU/CPU) and NumPy arrays.
    """
    # ---------------------------------------------------------
    # BRANCH 1: PyTorch Tensor Input (GPU/CPU)
    # ---------------------------------------------------------
    if isinstance(posterior, torch.Tensor):
        n_bins = posterior.shape[0]
        window_size = int(n_bins * window_fraction)
        
        # 1. Reduce Time Dimension (Max)
        # Note: torch.max returns (values, indices), we need [0] for values
        # PyTorch uses 'dim', not 'axis'
        spatial_profile = torch.max(posterior, dim=1)[0] 
        
        # 2. Circular Sliding Window Sum (using Conv1d for speed on GPU)
        # Concatenate right side to handle wrapping
        extended = torch.cat([spatial_profile, spatial_profile[:window_size]])
        
        # Reshape for conv1d: (Batch, Channel, Length)
        input_tensor = extended.view(1, 1, -1)
        # Kernel: Summing window of ones
        kernel = torch.ones((1, 1, window_size), device=posterior.device)
        
        # Calculate sums
        # Output shape: (1, 1, N_bins + 1) -> select first N_bins
        window_sums = F.conv1d(input_tensor, kernel)[0, 0]
        window_sums = window_sums[:n_bins]
        
        # Find best window start
        best_start = torch.argmax(window_sums)
        
        # 3. Crop and Unwrap
        # Generate indices on device
        indices = torch.arange(window_size, device=posterior.device) + best_start
        indices = indices % n_bins # Handle wrap-around
        
        return posterior[indices, :]

    # ---------------------------------------------------------
    # BRANCH 2: NumPy Array Input (CPU)
    # ---------------------------------------------------------
    else:
        n_bins = posterior.shape[0]
        window_size = int(n_bins * window_fraction)
        
        # 1. Reduce Time (NumPy uses 'axis')
        spatial_profile = np.max(posterior, axis=1)

        # 2. Circular Sliding Sum
        extended_profile = np.concatenate([spatial_profile, spatial_profile[:window_size]])
        window_sums = np.convolve(extended_profile, np.ones(window_size), mode='valid')
        
        best_start_idx = np.argmax(window_sums)

        # 3. Crop
        raw_indices = np.arange(best_start_idx, best_start_idx + window_size)
        wrapped_indices = raw_indices % n_bins
        
        return posterior[wrapped_indices, :]

class Decode1d:
    def __init__(
        self,
        neurons: core.Neurons,
        ratemap: core.Ratemap,
        epochs: Union[core.Epoch, None] = None,
        bin_size=0.5,
        slideby=None,
        n_jobs=1,
        mode='linear',
        no_spike_policy='keep',
    ):
        """1D decoding using ratemaps

        Parameters
        ----------
        neurons : core.Neurons
            neurons object containing spiketrains
        ratemap : core.Ratemap
            ratemap containing tuning curves
        epochs : core.Epoch, optional
            if provided then decode within these epochs only,if None then uses entire duration of neurons, by default None
        bin_size : float, optional
            binning size to calculate spike counts, by default 0.5
        slideby : float, optional
            slide the binning window by this amount, by default None
        n_jobs : int, optional
            number of parallel jobs for computation, by default 1
        mode : str, optional
            'linear' for linear track or 'circular' for circular track, by default 'linear'
        no_spike_policy : str, optional
            How to handle time bins with no spikes: 'keep', 'nan', or 'uniform', by default 'keep'
            - 'keep': Keep exp(-tau*frate) values
            - 'nan': Set posterior to NaN
            - 'uniform': Set to uniform distribution across positions
        """
        self.ratemap = ratemap
        self._events = None
        self.posterior = None
        self.neurons = neurons
        self.bin_size = bin_size
        self.pos_bin_size = ratemap.x_binsize
        self.decoded_position = None
        self.epochs = epochs
        self.slideby = slideby
        self.decoded_time = None
        self.n_spikes = None
        self.n_jobs = n_jobs
        self.mode = mode
        self.no_spike_policy = no_spike_policy

        self._estimate()

    def decode_cpu(self, spkcount, tuning_curves):
        """
        Bayesian decoding (CPU version) that handles both 1D and directional (2D) tuning curves.
        
        Parameters
        ----------
        spkcount : np.ndarray
            Shape (n_neurons, n_time_bins). Spike counts per bin.
        tuning_curves : np.ndarray
            Shape (n_neurons, n_spatial_bins) OR (n_neurons, n_spatial_bins, n_directions).

        Returns
        -------
        posterior : np.ndarray
            If directional: Shape (n_spatial_bins, n_time_bins, 2).
            If non-directional: Shape (n_spatial_bins, n_time_bins).
        """
        
        # ---------------------------------------------------------
        # PART 1: Pre-processing 
        # Flatten directional dimensions if necessary
        # ---------------------------------------------------------
        if tuning_curves.ndim == 3 and tuning_curves.shape[2] > 1:
            n_neurons, n_bins, n_dirs = tuning_curves.shape
            
            # Transpose to (Neurons, Dirs, Bins) -> Reshape to (Neurons, Total_Bins)
            # This ensures the first n_bins columns are Dir0, next n_bins are Dir1
            ratemaps = tuning_curves.transpose(0, 2, 1).reshape(n_neurons, -1)
            is_directional = True
        else:
            ratemaps = tuning_curves
            n_bins = tuning_curves.shape[1]
            is_directional = False

        # ---------------------------------------------------------
        # PART 2: Core Decoding
        # Formula: prob = (product(rates^spikes)) * exp(-tau * sum(rates))
        # ---------------------------------------------------------
        n_positions = ratemaps.shape[1]
        n_time_bins = spkcount.shape[1]
        prob = np.zeros((n_positions, n_time_bins))

        for i in range(n_positions):
            # Ignore neurons/indx which have zero frate at this location 
            # to avoid having frate product zero
            valid_indx = ratemaps[:, i] > 0
            
            if np.any(valid_indx):
                # (Rate ^ k)
                # Use newaxis to broadcast rates across time bins
                frate = (ratemaps[valid_indx, i, np.newaxis]) ** spkcount[valid_indx, :]
                
                # exp(-tau * sum(Rate))
                # The sum is over neurons for this specific position 'i'
                exp_frate = np.exp(-self.bin_size * np.sum(ratemaps[valid_indx, i]))
                
                # Combine: Product over neurons * Exponential term
                prob[i, :] = np.prod(frate, axis=0) * exp_frate

        # ---------------------------------------------------------
        # PART 3: Normalization
        # ---------------------------------------------------------
        old_settings = np.seterr(all="ignore")
        
        # Normalize across positions so sum(prob) = 1 for each time bin
        prob_sum = np.sum(prob, axis=0, keepdims=True)
        # Avoid division by zero
        prob /= prob_sum
        
        np.seterr(**old_settings)
        
        # Handle NaNs that might result from 0/0 division
        prob = np.nan_to_num(prob)

        # ---------------------------------------------------------
        # PART 4: Reshaping
        # ---------------------------------------------------------
        if is_directional:
            # raw_posterior is (2*n_bins, n_time)
            # Split into the two direction blocks based on how we stacked them in Part 1
            block0 = prob[:n_bins, :] # Direction 0
            block1 = prob[n_bins:, :] # Direction 1

            # Stack along a new 3rd dimension: (n_bins, n_time, 2)
            final_posterior = np.stack([block0, block1], axis=2)
            return final_posterior
        else:
            # Non-directional: Return as is (n_bins, n_time)
            return prob
        
    def decode_gpu(self, spkcount, tuning_curves):
        """
        Concatenated and GPU-optimized Bayesian Decoder.
        
        Parameters
        ----------
        spkcount : torch.Tensor
            Shape (n_neurons, n_time_bins)
        tuning_curves : torch.Tensor
            Shape (n_neurons, n_bins) OR (n_neurons, n_bins, n_dirs)
            
        Returns
        -------
        posterior : torch.Tensor
            Shape (n_bins, n_time) or (n_bins, n_time, 2)
        """
        
        # 1. Handle Tuning Curve Stacking
        # -----------------------------------------------------------------
        if tuning_curves.ndim == 3 and tuning_curves.shape[2] > 1:
            n_neurons, n_bins, n_dirs = tuning_curves.shape
            
            # Permute to (Neurons, Dirs, Bins) then flatten last two dims.
            # This ensures the first n_bins columns correspond to Dir 0, 
            # and the next n_bins correspond to Dir 1.
            stacked_tc = tuning_curves.permute(0, 2, 1).reshape(n_neurons, -1)
            is_directional = True
        else:
            stacked_tc = tuning_curves
            n_bins = tuning_curves.shape[1]
            is_directional = False

        # 2. Vectorized Decoding (replacing _decoder logic)
        # -----------------------------------------------------------------
        # We calculate in Log-Space to use matrix multiplication (much faster on GPU)
        # Formula: P(x|n) ~ (prod lambda^n) * exp(-tau * sum lambda)
        # Log Prob: sum(n * log(lambda)) - tau * sum(lambda)
        
        # Add tiny epsilon to avoid log(0) -> -inf
        epsilon = 1e-15 
        log_tc = torch.log(stacked_tc + epsilon)

        # Term 1: sum(n * log(lambda))
        # Matrix Mult: (Time, Neurons) @ (Neurons, TotalBins) -> (Time, TotalBins)
        term1 = torch.matmul(spkcount.T, log_tc)
        
        # Term 2: tau * sum(lambda)
        # Sum ratemaps across neurons for each position
        # Shape: (TotalBins,)
        term2 = self.bin_size * torch.sum(stacked_tc, dim=0)
        
        # Combine to get unnormalized log probability
        # Shape: (Time, TotalBins) - (TotalBins,) -> (Time, TotalBins) via broadcasting
        log_posterior = term1 - term2
        
        # Transpose to (TotalBins, Time) to match original format
        log_posterior = log_posterior.T

        # 3. Convert to Probability and Normalize
        # -----------------------------------------------------------------
        # Subtract max for numerical stability before exp (prevents overflow)
        # equivalent to the "softmax" trick
        log_posterior_stable = log_posterior - torch.max(log_posterior, dim=0, keepdim=True)[0]
        prob = torch.exp(log_posterior_stable)
        
        # Normalize columns to sum to 1
        # Add epsilon to denominator to prevent division by zero
        prob = prob / (torch.sum(prob, dim=0, keepdim=True) + epsilon)

        # 4. Reshape if Directional
        # -----------------------------------------------------------------
        if is_directional:
            # prob is (2*n_bins, n_time)
            # Split into the two direction blocks
            block0 = prob[:n_bins, :] # Direction 0
            block1 = prob[n_bins:, :] # Direction 1
            
            # Stack results: (n_bins, n_time, 2)
            final_posterior = torch.stack([block0, block1], dim=2)
            return final_posterior
        else:
            return prob

    def _estimate(self):
        """Estimates position with Position-wise Directional Reduction"""

        tuning_curves = self.ratemap.tuning_curves
        bincntr = self.ratemap.coords
        
        # Determine n_bins
        if tuning_curves.ndim == 3:
            n_bins = tuning_curves.shape[1]
            self.is_directional = True
        else:
            n_bins = tuning_curves.shape[1]
            self.is_directional = False

        # Handle Epochs or Single Session
        if self.epochs is not None:
            spkcount, nbins_list = self.neurons.get_spikes_in_epochs(
                self.epochs, self.bin_size, self.slideby
            )
            stacked_spkcount = np.hstack(spkcount)
            self.spkcount = spkcount
            self.nbins_epochs = nbins_list
            self.n_spikes = np.array([np.sum(spk) for spk in spkcount])
        else:
            binned = self.neurons.get_binned_spiketrains(self.bin_size)
            stacked_spkcount = binned.spike_counts
            nbins_list = [stacked_spkcount.shape[1]]
            self.n_spikes = np.sum(stacked_spkcount)

        # --- Call Helper Function ---
        raw_posterior = self.decode_cpu(
            stacked_spkcount, tuning_curves
        )

        # 1. Process Directionality & Marginalize
        # ---------------------------------------------------------
        if raw_posterior.ndim == 3:
            marginal_posterior = np.sum(raw_posterior, axis=2)
            
            prob_dir0 = np.sum(raw_posterior[:, :, 0], axis=0)
            prob_dir1 = np.sum(raw_posterior[:, :, 1], axis=0)
            dir_score = prob_dir0 - prob_dir1 
            is_directional = True
        else:
            marginal_posterior = raw_posterior
            dir_score = None
            is_directional = False

        # 2. Apply No-Spike Policy (To Marginal Posterior)
        # ---------------------------------------------------------
        if self.no_spike_policy != 'keep':
            no_spike_bins = np.sum(stacked_spkcount, axis=0) == 0
            
            if self.no_spike_policy == 'nan':
                marginal_posterior[:, no_spike_bins] = np.nan
                if is_directional:
                    dir_score[no_spike_bins] = np.nan
            elif self.no_spike_policy == 'uniform':
                n_states = marginal_posterior.shape[0]
                marginal_posterior[:, no_spike_bins] = 1.0 / n_states
                if is_directional:
                    dir_score[no_spike_bins] = 0.0

        # 3. Calculate Decoded Position (From Marginal)
        # ---------------------------------------------------------
        # FIX: Check for columns that are ALL NaNs before running argmax
        # If policy='nan', these columns will be pure NaNs and crash nanargmax
        all_nan_mask = np.all(np.isnan(marginal_posterior), axis=0)
        valid_mask = ~all_nan_mask
        
        # Initialize index array with 0 (safe default)
        max_indices = np.zeros(marginal_posterior.shape[1], dtype=np.int64)
        
        # Only compute argmax on valid columns
        if np.any(valid_mask):
            max_indices[valid_mask] = np.nanargmax(marginal_posterior[:, valid_mask], axis=0)
            
        decodedPos = bincntr[max_indices]

        # Set position to NaN where input was invalid (or per policy)
        if np.any(all_nan_mask):
            decodedPos[all_nan_mask] = np.nan
            
        if self.no_spike_policy != 'keep':
            decodedPos[no_spike_bins] = np.nan

        # 4. Split and Store
        # ---------------------------------------------------------
        if self.epochs is not None:
            cum_nbins = np.cumsum(nbins_list)[:-1]
            self.decoded_position = np.hsplit(decodedPos, cum_nbins)
            self.posterior = np.hsplit(marginal_posterior, cum_nbins)
            self.posterior_raw = np.hsplit(raw_posterior, cum_nbins)
            if is_directional:
                self.score = np.hsplit(dir_score, cum_nbins)
            
            self.decoded_time = []
            slideby = self.bin_size if self.slideby is None else self.slideby
            for i, n in enumerate(nbins_list):
                start = self.epochs[i].flatten()[0]
                self.decoded_time.append(start + np.arange(n)*slideby + self.bin_size/2)
        else:
            self.decoded_position = decodedPos
            self.posterior = marginal_posterior
            self.posterior_raw = raw_posterior
            if is_directional:
                self.score = dir_score
            t_start = self.neurons.t_start
            n = marginal_posterior.shape[1]
            slideby = self.bin_size if self.slideby is None else self.slideby
            self.decoded_time = t_start + np.arange(n)*slideby + self.bin_size/2

    def _get_jd(self, posteriors, jump_stat="mean"):
        """Calculate jump distance for posterior matrices"""
        return jump_distance(posteriors, jump_stat=jump_stat, mode=self.mode, norm=True)


    def get_wcorr(self, jump_stat=None, posteriors=None, mode=None, n_spikes_thresh=1, 
                  gpu=False, shuffle=False, n_iter=1000, shuffle_method="neuron_id", win_per = None,
                  return_shuffle=False):
        """
        Calculate weighted correlation for trajectories, with optional shuffling for statistics.
        """
        
        # --- 1. Setup & Defaults ---
        if posteriors is None:
            assert self.posterior is not None, "No posteriors found"
            posteriors = self.posterior
        
        if mode is None:
            mode = self.mode

        device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'
        if gpu and device == 'cpu':
            print("Warning: GPU requested but not available. Falling back to CPU.")

        # =========================================================
        # PHASE 1: Calculate Real Scores (Actual Data)
        # =========================================================
        
        # --- GPU Execution ---
        if gpu and device == 'cuda':
            scores_list = []
            # Loop through each epoch separately since time dimensions differ
            for epoch in tqdm(posteriors, desc="Weighted Corr (GPU)", leave=not shuffle):
                if isinstance(epoch, torch.Tensor):
                    epoch_gpu = epoch.to(device, non_blocking=True)
                else:
                    epoch_gpu = torch.tensor(epoch, device=device, dtype=torch.float32)

                # Handle 3D posteriors (sum directions) for wcorr
                if epoch_gpu.ndim == 3:
                    epoch_gpu = torch.sum(epoch_gpu, dim=2)

                score = opt_wcorr(epoch_gpu, mode=mode)
                scores_list.append(score.cpu().numpy())
            real_scores = np.array(scores_list)

        # --- CPU Execution ---
        else:
            # Handle potential 3D posteriors for CPU path
            cpu_posteriors = [p if p.ndim == 2 else np.sum(p, axis=2) for p in posteriors]
            
            real_scores = Parallel(n_jobs=self.n_jobs)(
                delayed(wcorr)(_, mode=mode) for _ in cpu_posteriors
            )
            real_scores = np.array(real_scores)
        
        # --- Apply Spike Threshold Mask ---
        if n_spikes_thresh > 0 and self.n_spikes is not None:
            if isinstance(self.n_spikes, np.ndarray):
                low_spike_mask = self.n_spikes < n_spikes_thresh
                real_scores[low_spike_mask] = np.nan
            elif self.n_spikes < n_spikes_thresh:
                real_scores[:] = np.nan

        # --- Calculate Jump Stats if requested ---
        real_jd = self._get_jd(posteriors, jump_stat) if jump_stat is not None else None

        # =========================================================
        # PHASE 2: Return Early if No Shuffling
        # =========================================================
        if not shuffle:
            if real_jd is not None:
                return real_scores, real_jd
            return real_scores

        # =========================================================
        # PHASE 3: Shuffling & P-Values
        # =========================================================
        
        # Pre-allocate Result Arrays (CPU)
        n_epochs = len(posteriors)
        shuffle_scores = np.zeros((n_iter, n_epochs), dtype=np.float32)

        # --- A. GPU Shuffle Strategy (OPTIMIZED) ---
        if gpu and device == 'cuda':
            
            # ---------------------------------------------------------
            # STRATEGY 1: Neuron ID Shuffle (Re-decoding required)
            # ---------------------------------------------------------
            if shuffle_method == "neuron_id":
                # 1. Prepare Static Data on GPU
                spk_list = [torch.tensor(s, device=device, dtype=torch.float32) for s in self.spkcount]
                full_spk_gpu = torch.cat(spk_list, dim=1) 
                
                # Tuning Curves
                tc_gpu = torch.tensor(self.ratemap.tuning_curves, device=device, dtype=torch.float32)
                n_neurons = tc_gpu.shape[0]
                
                # Get epoch lengths for splitting the posterior later
                epoch_lengths = [p.shape[-1] if hasattr(p, 'shape') else len(p[0]) for p in posteriors] 

                # 2. The Shuffle Loop
                for i in tqdm(range(n_iter), desc="Shuffling (Neuron ID | GPU)"):
                    
                    # a. Shuffle Tuning Curves on GPU
                    perm_idx = torch.randperm(n_neurons, device=device)
                    shuffled_tc = tc_gpu[perm_idx]

                    # b. Decode Everything (using decode_gpu)
                    full_posterior = self.decode_gpu(full_spk_gpu, shuffled_tc)
                    
                    # c. Handle Directionality (Reduce to 2D)
                    if full_posterior.ndim == 3:
                        full_posterior = torch.sum(full_posterior, dim=2)
                    
                    # d. Split back into Epochs
                    epoch_posteriors = torch.split(full_posterior, epoch_lengths, dim=1)
                    
                    # e. Calculate wcorr for each epoch
                    for ep_idx, epoch_img in enumerate(epoch_posteriors):
                        s = opt_wcorr(epoch_img, mode=mode)
                        shuffle_scores[i, ep_idx] = s.item()

            # ---------------------------------------------------------
            # STRATEGY 2: Column Cycle Shuffle (Posterior Shifting)
            # ---------------------------------------------------------
            elif shuffle_method == "column_cycle":
                
                gpu_posteriors = []
                
                # We only need to track new scores if we are actually cropping
                if win_per is not None:
                    cropped_real_scores = [] 

                for p in posteriors:
                    # Load to GPU
                    t_p = torch.tensor(p, device=device, dtype=torch.float32) if not isinstance(p, torch.Tensor) else p.to(device)
                    
                    # Handle 3D
                    if t_p.ndim == 3:
                        t_p = torch.sum(t_p, dim=2)
                    
                    # >>> CONDITIONAL LOGIC <<<
                    if win_per is not None:
                        # Case A: Crop is ACTIVE
                        # 1. Apply Crop
                        t_p_final = crop_posterior(t_p, window_fraction=win_per)
                        
                        # 2. Re-calculate Real Score (Must compare Cropped vs Cropped)
                        s_real = opt_wcorr(t_p_final, mode=mode).item()
                        cropped_real_scores.append(s_real)
                    else:
                        # Case B: Crop is INACTIVE (Standard)
                        # Use full matrix
                        t_p_final = t_p
                        # Do NOT re-calculate real_score; Phase 1 score is already correct for full data

                    gpu_posteriors.append(t_p_final)

                # Overwrite global real_scores only if we cropped
                if win_per is not None:
                    real_scores = np.array(cropped_real_scores)

                # 2. The Shuffle Loop (Works for both Cropped and Full data)
                for i in tqdm(range(n_iter), desc="Shuffling (Column Cycle | GPU)"):
                    for ep_idx, epoch_img in enumerate(gpu_posteriors):
                        n_time = epoch_img.shape[1]
                        
                        # Handle too small matrices
                        if n_time < 2:
                            shuffle_scores[i, ep_idx] = opt_wcorr(epoch_img, mode=mode).item()
                            continue

                        # Generate random shift
                        shift = int(torch.randint(low=1, high=n_time, size=(1,)).item())
                        
                        # Apply circular roll along time axis
                        shuffled_img = torch.roll(epoch_img, shifts=shift, dims=1)
                        
                        # Calculate Score
                        s = opt_wcorr(shuffled_img, mode=mode)
                        shuffle_scores[i, ep_idx] = s.item()

            else:
                raise ValueError(f"Unknown GPU shuffle method: {shuffle_method}")

        # --- B. Legacy CPU/Generic Shuffle Strategy ---
        else:
            # We recursively call this function but with shuffle=False to get scores
            shuffle_result = self._shuffler(
                func=self.get_wcorr,
                n_iter=n_iter,
                method=shuffle_method,
                gpu=False,
                # Pass args to ensure recursive call stops at Phase 1
                shuffle=False,
                mode=mode,
                jump_stat=jump_stat,
                n_spikes_thresh=n_spikes_thresh 
            )
            
            # Handle formatting differences if jump stats were returned in the recursive calls
            if real_jd is not None:
                # shuffle_result is likely a list of tuples [(score, jd), ...]
                shuffle_result = np.array(shuffle_result, dtype=object)
                shuffle_scores = np.array([x[0] for x in shuffle_result])
            else:
                shuffle_scores = np.array(shuffle_result)

        # --- Calculate P-values ---
        p_values = np.zeros(len(real_scores))
        for i in range(len(real_scores)):
            if not np.isnan(real_scores[i]):
                # One-tailed check: direction depends on the sign of the real score
                if real_scores[i] > 0:
                    # Positive correlation: Test if shuffle is greater than or equal to real
                    p_values[i] = np.mean(shuffle_scores[:, i] >= real_scores[i])
                else:
                    # Negative correlation: Test if shuffle is less than or equal to real
                    p_values[i] = np.mean(shuffle_scores[:, i] <= real_scores[i])
            else:
                p_values[i] = np.nan

        # =========================================================
        # PHASE 4: Final Return
        # =========================================================
        
        # 1. Start with the unpacked real results
        ret_vals = [real_scores]
        
        # 2. Add Jump Distance if it exists
        if real_jd is not None:
            ret_vals.append(real_jd)
        
        # 3. Append Shuffle Items (if shuffle was requested)
        if shuffle:
            # If user explicitly asked for the raw shuffle matrix
            if return_shuffle:
                ret_vals.append(shuffle_scores)
                
            # P-values always go at the very end
            ret_vals.append(p_values)
        
        # 4. Return
        if len(ret_vals) == 1:
            return ret_vals[0]
        return tuple(ret_vals)

    def get_radon_transform(self, nlines=5000, margin=0.1, jump_stat=None, posteriors=None, 
                            mode=None, n_spikes_thresh=1, gpu=False, 
                            shuffle=False, n_iter=1000, shuffle_method="neuron_id", 
                            return_shuffle=False):
        """
        Calculate radon transform scores, velocities, and intercepts, with optional shuffling.
        """
        
        # --- 1. Setup & Defaults ---
        if posteriors is None:
            assert self.posterior is not None, "No posteriors found"
            posteriors = self.posterior
        
        if mode is None:
            mode = self.mode
        
        # Calculate smoothing radius in bins
        neighbours = int(margin / self.ratemap.x_binsize)
        
        device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'
        if gpu and device == 'cpu':
            print("Warning: GPU requested but not available. Falling back to CPU.")

        # =========================================================
        # PHASE 1: Calculate Real Scores (Actual Data)
        # =========================================================
        
        # --- GPU Execution ---
        if gpu and device == 'cuda':
            results = []
            epoch_lengths = [] # Needed for the shuffle split later
            
            # Loop through each epoch separately since time dimensions differ
            for epoch in tqdm(posteriors, desc="Radon Transform (GPU)", leave=not shuffle):
                # Move input to GPU
                if isinstance(epoch, torch.Tensor):
                    images_gpu = epoch.to(device, non_blocking=True)
                else:
                    images_gpu = torch.tensor(epoch, device=device, dtype=torch.float32)

                # Store length for the shuffle phase
                epoch_lengths.append(images_gpu.shape[1])

                # Handle 3D posteriors (sum directions)
                if images_gpu.ndim == 3:
                    images_gpu = torch.sum(images_gpu, dim=2)

                # Call compiled function
                res = opt_radon_transform(
                    images=images_gpu,
                    n_lines=nlines,
                    dt=float(self.bin_size),
                    dx=float(self.pos_bin_size),
                    smoothing_radius=neighbours,
                    mode=mode
                )

                # Move output to CPU immediately
                res_cpu = tuple(r.cpu().item() for r in res)
                results.append(res_cpu)
                        
            real_score, real_velocity, real_intercept = np.asarray(results).T

        # --- CPU Execution ---
        else:
            epoch_lengths = [p.shape[1] for p in posteriors] # Needed for consistency
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(radon_transform)(
                    image=epoch if epoch.ndim == 2 else np.sum(epoch, axis=2),
                    n_lines=nlines,
                    dt=self.bin_size,
                    dx=self.pos_bin_size,
                    smoothing_radius=neighbours,
                    mode=mode
                )
                for epoch in posteriors
            )
            real_score, real_velocity, real_intercept = np.asarray(results).T
        
        # --- Apply Spike Threshold Mask ---
        if n_spikes_thresh > 0 and self.n_spikes is not None:
            if isinstance(self.n_spikes, np.ndarray):
                low_spike_mask = self.n_spikes < n_spikes_thresh
                real_score[low_spike_mask] = np.nan
                real_velocity[low_spike_mask] = np.nan
                real_intercept[low_spike_mask] = np.nan
            elif self.n_spikes < n_spikes_thresh:
                real_score[:] = np.nan
                real_velocity[:] = np.nan
                real_intercept[:] = np.nan
                
        # --- Calculate Jump Stats if requested ---
        real_jd = self._get_jd(posteriors, jump_stat) if jump_stat is not None else None

        # =========================================================
        # PHASE 2: Return Early if No Shuffling
        # =========================================================
        if not shuffle:
            if real_jd is not None:
                return real_score, real_velocity, real_intercept, real_jd
            return real_score, real_velocity, real_intercept

        # =========================================================
        # PHASE 3: Shuffling & P-Values
        # =========================================================
        
        # --- A. GPU Shuffle Strategy (OPTIMIZED) ---
        if gpu and device == 'cuda':
            
            # Pre-allocate Result Arrays (CPU)
            n_epochs = len(posteriors)
            shuffle_scores = np.zeros((n_iter, n_epochs), dtype=np.float32)
            
            if return_shuffle:
                shuffle_vels = np.zeros((n_iter, n_epochs), dtype=np.float32)
                shuffle_ints = np.zeros((n_iter, n_epochs), dtype=np.float32)

            # ---------------------------------------------------------
            # STRATEGY 1: Neuron ID Shuffle (Re-decoding required)
            # ---------------------------------------------------------
            if shuffle_method == "neuron_id":
                
                # 1. Prepare Static Data on GPU
                spk_list = [torch.tensor(s, device=device, dtype=torch.float32) for s in self.spkcount]
                full_spk_gpu = torch.cat(spk_list, dim=1) 
                
                # Tuning Curves
                tc_gpu = torch.tensor(self.ratemap.tuning_curves, device=device, dtype=torch.float32)
                n_neurons = tc_gpu.shape[0]

                # 2. The Shuffle Loop
                for i in tqdm(range(n_iter), desc="Shuffling (Neuron ID | GPU)"):
                    
                    # a. Shuffle Tuning Curves on GPU
                    perm_idx = torch.randperm(n_neurons, device=device)
                    shuffled_tc = tc_gpu[perm_idx]

                    # b. Decode Everything (using decode_gpu)
                    full_posterior = self.decode_gpu(full_spk_gpu, shuffled_tc)

                    # c. Handle Directionality (Reduce to 2D)
                    if full_posterior.ndim == 3:
                        full_posterior = torch.sum(full_posterior, dim=2)
                    
                    # d. Split back into Epochs
                    epoch_posteriors = torch.split(full_posterior, epoch_lengths, dim=1)
                    
                    # e. Calculate Radon for each epoch
                    for ep_idx, epoch_img in enumerate(epoch_posteriors):
                        s, v, intc = opt_radon_transform(
                            epoch_img, 
                            n_lines=nlines, 
                            dt=float(self.bin_size), 
                            dx=float(self.pos_bin_size), 
                            smoothing_radius=neighbours, 
                            mode=mode
                        )
                        
                        shuffle_scores[i, ep_idx] = s.item()
                        if return_shuffle:
                            shuffle_vels[i, ep_idx] = v.item()
                            shuffle_ints[i, ep_idx] = intc.item()
            
            # ---------------------------------------------------------
            # STRATEGY 2: Column Cycle Shuffle (Posterior Shifting)
            # ---------------------------------------------------------
            elif shuffle_method == "column_cycle":
                
                # 1. Prepare Posteriors on GPU
                gpu_posteriors = []
                for p in posteriors:
                    t_p = torch.tensor(p, device=device, dtype=torch.float32) if not isinstance(p, torch.Tensor) else p.to(device)
                    # Handle 3D
                    if t_p.ndim == 3:
                        t_p = torch.sum(t_p, dim=2)
                    gpu_posteriors.append(t_p)

                # 2. The Shuffle Loop
                for i in tqdm(range(n_iter), desc="Shuffling (Column Cycle | GPU)"):
                    for ep_idx, epoch_img in enumerate(gpu_posteriors):
                        n_time = epoch_img.shape[1]
                        if n_time < 2:
                            # Cannot shift 1-bin epoch meaningfully
                            s, v, intc = opt_radon_transform(
                                epoch_img, n_lines=nlines, dt=float(self.bin_size), 
                                dx=float(self.pos_bin_size), smoothing_radius=neighbours, mode=mode
                            )
                        else:
                            # Generate random shift
                            shift = int(torch.randint(low=1, high=n_time, size=(1,)).item())
                            
                            # Apply circular roll along time axis (dim 1)
                            shuffled_img = torch.roll(epoch_img, shifts=shift, dims=1)
                            
                            s, v, intc = opt_radon_transform(
                                shuffled_img, n_lines=nlines, dt=float(self.bin_size), 
                                dx=float(self.pos_bin_size), smoothing_radius=neighbours, mode=mode
                            )
                        
                        shuffle_scores[i, ep_idx] = s.item()
                        if return_shuffle:
                            shuffle_vels[i, ep_idx] = v.item()
                            shuffle_ints[i, ep_idx] = intc.item()
            
            else:
                raise ValueError(f"Unknown GPU shuffle method: {shuffle_method}")

            # Pack results if requested (common to both GPU strategies)
            if return_shuffle:
                shuffle_results_packed = np.stack([shuffle_scores, shuffle_vels, shuffle_ints], axis=2)

        # --- B. Legacy CPU Shuffle Strategy ---
        else:
            # Recursively call this function with shuffle=False
            shuffle_raw = self._shuffler(
                func=self.get_radon_transform,
                n_iter=n_iter,
                method=shuffle_method,
                gpu=False,
                # Recursive Args
                shuffle=False,
                nlines=nlines,
                margin=margin,
                mode=mode,
                jump_stat=jump_stat,
                n_spikes_thresh=n_spikes_thresh
            )
            
            shuffle_raw = np.array(shuffle_raw)
            shuffle_scores = shuffle_raw[:, 0, :]
            
            if return_shuffle:
                shuffle_results_packed = np.transpose(shuffle_raw, (0, 2, 1))

        # --- Calculate P-values ---
        p_values = np.zeros(len(real_score))
        
        # Vectorized comparison where possible
        valid_mask = ~np.isnan(real_score)
        if np.any(valid_mask):
            # (n_iter, n_valid) >= (n_valid,)
            hits = shuffle_scores[:, valid_mask] >= real_score[valid_mask]
            p_values[valid_mask] = np.mean(hits, axis=0)
            
        p_values[~valid_mask] = np.nan

        # =========================================================
        # PHASE 4: Final Return
        # =========================================================
        
        ret_vals = [real_score, real_velocity, real_intercept]
        
        if real_jd is not None:
            ret_vals.append(real_jd)
        
        if shuffle:
            if return_shuffle:
                ret_vals.append(shuffle_results_packed)
            ret_vals.append(p_values)
        
        return tuple(ret_vals)

    def _shuffler(self, func, n_iter, method, gpu=False, **kwargs):
        """Legacy CPU shuffler (unchanged)"""
        assert callable(func), "scoring function is not callable"
        cum_nbins = np.cumsum(self.nbins_epochs)[:-1]
        stacked_posterior = np.hstack(self.posterior)
        spkcount = np.hstack(self.spkcount)
        
        def _single_shuffle(seed):
            np.random.seed(seed)
            if method == "neuron_id":
                shuffled_tc = self.ratemap.tuning_curves.copy()
                np.random.default_rng(seed).shuffle(shuffled_tc)
                raw_posterior = self.decode_cpu(spkcount, shuffled_tc)
                marginal_posterior = np.sum(raw_posterior, axis=2)
                shuffle_posteriors = np.hsplit(marginal_posterior, cum_nbins)
            elif method == "column_cycle":
                shuffle_posteriors = np.hsplit(column_shift(stacked_posterior), cum_nbins)
            return func(posteriors=shuffle_posteriors, gpu=gpu, **kwargs)
        
        score = Parallel(n_jobs=self.n_jobs)(
            delayed(_single_shuffle)(i) for i in tqdm(range(n_iter), desc="Shuffling (CPU)")
        )
        return np.array(score)

    def plot_summary(
        self, 
        scores=None,
        velocities=None,
        intercepts=None,
        method='radon_transform',
        n_examples=5,
        prob_cmap="hot",
        count_cmap="binary",
        line_color="#00E676"
    ):
        """Plot summary of decoded events
        
        Parameters
        ----------
        scores : ndarray, optional
            Scores for each event (required if method='radon_transform')
        velocities : ndarray, optional
            Velocities for each event (required if method='radon_transform')
        intercepts : ndarray, optional
            Intercepts for each event (required if method='radon_transform')
        method : str, optional
            Plotting method: 'radon_transform' or 'wcorr', by default 'radon_transform'
        n_examples : int, optional
            Number of example events to plot, by default 5
        prob_cmap : str, optional
            Colormap for posterior, by default 'hot'
        count_cmap : str, optional
            Colormap for spike counts, by default 'binary'
        line_color : str, optional
            Color for radon line overlay, by default '#00E676'
            
        Examples
        --------
        >>> score, velocity, intercept = decoder.get_radon_transform(nlines=5000)
        >>> decoder.plot_summary(
        ...     scores=score,
        ...     velocities=velocity,
        ...     intercepts=intercept,
        ...     method='radon_transform'
        ... )
        """
        if method == "radon_transform":
            if scores is None or velocities is None or intercepts is None:
                raise ValueError(
                    "For radon_transform plotting, must provide scores, velocities, and intercepts"
                )
            self._plot_radon_transform(
                scores=scores,
                velocities=velocities,
                intercepts=intercepts,
                n_examples=n_examples,
                prob_cmap=prob_cmap,
                count_cmap=count_cmap,
                lc=line_color
            )
        elif method == "wcorr":
            if scores is None:
                raise ValueError("For wcorr plotting, must provide scores")
            self._plot_wcorr(
                scores=scores,
                n_examples=n_examples,
                prob_cmap=prob_cmap,
                count_cmap=count_cmap
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'radon_transform' or 'wcorr'")

    def _plot_wcorr(
        self, 
        scores,
        n_examples=5,
        prob_cmap="hot", 
        count_cmap="binary"
    ):
        """Internal plotting method for wcorr results"""
        n_posteriors = len(self.posterior)
        n_examples = min(n_examples, n_posteriors)
        posterior_ind = np.random.default_rng().integers(0, n_posteriors, n_examples)
        arrs = [self.posterior[i] for i in posterior_ind]

        _, axs = plt.subplots(3, n_examples, sharey='row', sharex='col', 
                             figsize=[2.2*n_examples, 8])
        if n_examples == 1:
            axs = axs[:, np.newaxis]

        # Sort neurons by their peak firing location
        zsc_tuning = stats.zscore(self.ratemap.tuning_curves, axis=1)
        sort_ind = np.argsort(np.argmax(zsc_tuning, axis=1))
        n_neurons = self.neurons.n_neurons

        for i, arr in enumerate(arrs):
            t_start = self.epochs[posterior_ind[i]].flatten()[0]
            t_stop = self.epochs[posterior_ind[i]].flatten()[1]
            score = scores[posterior_ind[i]]

            arr_smooth = np.apply_along_axis(
                np.convolve, axis=0, arr=arr, v=np.ones(2 * 2 + 1), mode='same'
            )
            t = np.arange(arr.shape[1]) * self.bin_size + t_start
            pos = np.arange(arr.shape[0]) * self.pos_bin_size

            axs[0, i].pcolormesh(t, pos, arr_smooth, cmap=prob_cmap)
            axs[0, i].set_ylim([pos.min(), pos.max()])
            axs[0, i].set_title(f"#{posterior_ind[i]}\nwcorr={score:.2f}")

            axs[1, i].pcolormesh(
                t,
                np.arange(n_neurons),
                self.spkcount[posterior_ind[i]],
                cmap=count_cmap,
            )
            
            # Plot raster without sorting (or sort by peak in decoded neurons only)
            plotting.plot_raster(
                self.neurons.time_slice(t_start=t_start, t_stop=t_stop),
                ax=axs[2, i],
                color="k",
            )
            if i == 0:
                axs[0, i].set_ylabel("Position (cm)")
                axs[1, i].set_ylabel("Neurons")
                axs[2, i].set_ylabel("Neurons")
            else:
                axs[2, i].set_ylabel("")
                
        axs[-1, n_examples//2].set_xlabel("Time (s)")
        plt.tight_layout()

    def _plot_radon_transform(
        self, 
        scores,
        velocities,
        intercepts,
        n_examples=5,
        prob_cmap="RdBu_r",
        count_cmap="binary", 
        lc="#00E676",
        mode=None
    ):
        """Internal plotting method for radon transform results
        
        Parameters
        ----------
        scores : array
            Radon transform scores
        velocities : array
            Velocities in cm/s
        intercepts : array
            Intercepts
        n_examples : int, optional
            Number of examples to plot, by default 5
        prob_cmap : str, optional
            Colormap for posterior, by default "RdBu_r"
        count_cmap : str, optional
            Colormap for spike counts, by default "binary"
        lc : str, optional
            Line color for trajectory, by default "#00E676"
        mode : str, optional
            'linear' or 'circular', defaults to self.mode
        """
        if mode is None:
            mode = self.mode
        
        n_posteriors = len(self.posterior)
        n_examples = min(n_examples, n_posteriors)
        posterior_ind = np.random.default_rng().integers(0, n_posteriors, n_examples)
        arrs = [self.posterior[i] for i in posterior_ind]
        
        fig, axs = plt.subplots(4, n_examples, sharey='row', sharex='col', 
                                figsize=[2.2*n_examples, 10])
        if n_examples == 1:
            axs = axs[:, np.newaxis]
        
        n_neurons = self.neurons.n_neurons
        
        # Get position info from ratemap (constant across events)
        pos_coords = self.ratemap.x_coords() if callable(self.ratemap.x_coords) else self.ratemap.x_coords
        pos_min, pos_max = pos_coords[0], pos_coords[-1]
        
        for i, arr in enumerate(arrs):
            t_start = self.epochs[posterior_ind[i]].flatten()[0]
            t_stop = self.epochs[posterior_ind[i]].flatten()[1]
            score = scores[posterior_ind[i]]
            velocity = velocities[posterior_ind[i]]
            intercept = intercepts[posterior_ind[i]]
            
            # Smooth for visualization
            arr_smooth = np.apply_along_axis(
                np.convolve, axis=0, arr=arr, v=np.ones(2 * 2 + 1), mode='same'
            )
            
            n_pos_bins = arr_smooth.shape[0]
            n_time_bins = arr_smooth.shape[1]
            
            # Position edges: use constant pos_bin_size
            pos_edges = np.arange(n_pos_bins + 1) * self.pos_bin_size + pos_min - self.pos_bin_size/2
            
            # Time edges: calculate from event duration (varies per event)
            actual_time_binsize = (t_stop - t_start) / n_time_bins
            t_edges = np.linspace(t_start - actual_time_binsize/2,
                                t_stop + actual_time_binsize/2,
                                n_time_bins + 1)
            
            # Centers for line overlay
            t_centers = np.linspace(t_start, t_stop, n_time_bins)
            
            # Calculate trajectory line
            trajectory = velocity * (t_centers - t_start) + intercept
            
            # Handle circular wrapping for trajectory
            if mode == 'circular':
                track_length = pos_max - pos_min
                trajectory = np.mod(trajectory - pos_min, track_length) + pos_min
            
            # Plot 1: Posterior without margin
            axs[0, i].pcolormesh(t_edges, pos_edges, arr_smooth, cmap=prob_cmap, shading='flat')
            axs[0, i].plot(t_centers, trajectory, color=lc, lw=2)
            axs[0, i].set_ylim([pos_min, pos_max])
            axs[0, i].set_title(f"#{posterior_ind[i]}\ns={score:.2f}, v={velocity:.1f} cm/s")
            
            # Plot 2: Posterior with margin
            margin_bins = 2
            arr_margin = np.apply_along_axis(
                np.convolve, axis=0, arr=arr_smooth, 
                v=np.ones(2 * margin_bins + 1), mode="same"
            )
            axs[1, i].pcolormesh(t_edges, pos_edges, arr_margin, cmap=prob_cmap, shading='flat')
            axs[1, i].plot(t_centers, trajectory, color=lc, lw=2)
            axs[1, i].set_ylim([pos_min, pos_max])
            
            # Plot 3: Spike counts
            neuron_edges = np.arange(n_neurons + 1) - 0.5
            axs[2, i].pcolormesh(t_edges, neuron_edges, self.spkcount[posterior_ind[i]], 
                                cmap=count_cmap, shading='flat')
            
            # Plot 4: Raster
            plotting.plot_raster(
                self.neurons.time_slice(t_start=t_start, t_stop=t_stop),
                ax=axs[3, i], color="k"
            )
            
            if i == 0:
                axs[0, i].set_ylabel("Position")
                axs[1, i].set_ylabel("Position\n+ margin")
                axs[2, i].set_ylabel("Neurons")
                axs[3, i].set_ylabel("Neurons")
            else:
                axs[3, i].set_ylabel("")
        
        axs[-1, n_examples//2].set_xlabel("Time (s)")
        plt.tight_layout()

class Decode2d:
    pass