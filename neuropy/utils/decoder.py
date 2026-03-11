import numpy as np
import torch
import torch.nn.functional as F


def radon_transform(
    image: np.ndarray, 
    n_lines: int = 10000, 
    dt: float = 1.0, 
    dx: float = 1.0, 
    smoothing_radius: int = 1, 
    mode: str = 'linear'
):
    """
    Performs a Radon-like transform to find the best linear trajectory.
    
    In 'circular' mode, the image is first aligned by calculating the circular 
    mean of the position axis and shifting it to the center, then processed 
    as a linear image (preserving sign/slope direction).

    Parameters:
        image: 2D numpy array (rows=position, cols=time).
        n_lines: Number of random lines to sample.
        dt: Temporal resolution.
        dx: Spatial resolution.
        smoothing_radius: Radius for smoothing along position.
        mode: 'linear' or 'circular'.

    Returns:
        score: Mean intensity along the best line.
        velocity: Estimated velocity (physical units).
        intercept: Estimated intercept (physical units).
    """
    n_rows, n_cols = image.shape
    
    # 1. Pre-process image (Smooth along position axis)
    # -----------------------------------------------------------
    kernel = np.ones(2 * smoothing_radius + 1)
    smoothed = image.copy()

    if smoothing_radius > 0:
        if mode == 'circular':
            # Pad top/bottom with wrapping for correct circular smoothing
            padded = np.pad(image, ((smoothing_radius, smoothing_radius), (0, 0)), mode='wrap')
            for t in range(n_cols):
                smoothed[:, t] = np.convolve(padded[:, t], kernel, mode='valid')
        else:
            # Standard zero-padded convolution
            for t in range(n_cols):
                smoothed[:, t] = np.convolve(image[:, t], kernel, mode='same')

    # 2. Handle Circular Shift Alignment
    # -----------------------------------------------------------
    shift_pixel_amount = 0
    
    if mode == 'circular':
        # --- A. Calculate Bin Centers ---
        # Create edges from 0 to 2*pi
        y_edges = np.linspace(0, 2 * np.pi, n_rows + 1, endpoint=True)
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        y_rad = y_centers[:, None] # Shape (n_rows, 1)

        # --- B. Calculate Global Circular Mean Phase ---
        # Weighted sum of sin and cos using the whole smoothed image
        sum_sin = np.nansum(smoothed * np.sin(y_rad))
        sum_cos = np.nansum(smoothed * np.cos(y_rad))
        
        mean_phase = np.arctan2(sum_sin, sum_cos)
        # Map phase back to index space [0, n_rows)
        mean_idx = ((mean_phase + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi) * n_rows
        
        # --- C. Shift to Center ---
        center_target = n_rows // 2
        shift_pixel_amount = int(np.round(center_target - mean_idx))
        
        # Roll the smoothed image so the "mass" is centered
        working_image = np.roll(smoothed, shift_pixel_amount, axis=0)
    else:
        working_image = smoothed

    # 3. Generate Random Trajectory Parameters
    # -----------------------------------------------------------
    rng = np.random.default_rng()
    
    # Threshold to avoid vertical lines
    min_phi = np.arctan2(1, n_rows)
    
    # Oversample phi to filter out invalid angles efficiently
    phi_candidates = rng.uniform(-np.pi / 2, np.pi / 2, int(n_lines * 1.5))
    valid_mask = np.abs(phi_candidates) >= min_phi
    phi = phi_candidates[valid_mask][:n_lines]
    
    # Fill remaining if filtered too many
    if len(phi) < n_lines:
        extra_needed = n_lines - len(phi)
        extra_phi = rng.uniform(-np.pi / 2, np.pi / 2, extra_needed * 2)
        phi = np.concatenate([phi, extra_phi[np.abs(extra_phi) >= min_phi]])[:n_lines]

    # Generate rho (distance from center)
    diag_len = np.sqrt((n_cols - 1)**2 + (n_rows - 1)**2)
    rho = rng.uniform(-diag_len / 2, diag_len / 2, n_lines)

    # 4. Compute Line Trajectories (Linear Logic)
    # -----------------------------------------------------------
    # We now treat 'working_image' as a linear map (even if originally circular),
    # because we have centered the data.
    
    t_mid_idx = (n_cols - 1) / 2
    y_mid_idx = (n_rows - 1) / 2
    
    t_indices = np.arange(n_cols)
    sin_phi = np.sin(phi)[:, None]
    cos_phi = np.cos(phi)[:, None]
    rho_bc = rho[:, None]

    # Calculate Y coordinates for every time point t
    y_lines_float = (rho_bc - (t_indices - t_mid_idx) * cos_phi) / sin_phi + y_mid_idx
    y_lines = np.rint(y_lines_float).astype(np.int64)

    # 5. Extract Values (Radon Projection)
    # -----------------------------------------------------------
    t_grid = np.broadcast_to(t_indices, y_lines.shape)
    
    # Use standard linear extraction strategy: 
    # Fill out-of-bounds indices with the column median to avoid edge artifacts
    col_medians = np.nanmedian(working_image, axis=0)
    posterior = np.tile(col_medians, (n_lines, 1))
    
    # Overwrite valid indices with actual image data
    valid_mask = (y_lines >= 0) & (y_lines < n_rows)
    posterior[valid_mask] = working_image[y_lines[valid_mask], t_grid[valid_mask]]

    # 6. Find Best Line
    # -----------------------------------------------------------
    line_scores = np.nanmean(posterior, axis=1)
    best_idx = np.argmax(line_scores)
    
    best_score = line_scores[best_idx]
    best_phi = phi[best_idx]
    best_rho = rho[best_idx]

    # 7. Convert to Physical Units
    # -----------------------------------------------------------
    phys_t_mid = n_cols * dt / 2.0
    phys_p_mid = n_rows * dx / 2.0
    
    best_tan = np.tan(best_phi)
    best_sin = np.sin(best_phi)

    # Velocity: Slope is invariant to Y-shift
    velocity = -dx / (dt * best_tan)
    
    # Intercept: Needs to be corrected if we shifted the image
    # The calculated intercept is relative to the *shifted* image.
    # We subtract the physical shift to map it back to the original coordinate system.
    # Note: This returns a linear intercept. For a wrapped line, this points to 
    # the intercept of the "unwrapped" continuous line segment.
    raw_intercept = (
        (dx * phys_t_mid) / (dt * best_tan) 
        + (best_rho / best_sin) * dx 
        + phys_p_mid
    )
    
    intercept = raw_intercept - (shift_pixel_amount * dx)
    
    return best_score, velocity, intercept


@torch.inference_mode()
def radon_transform_batch(
    images: torch.Tensor, 
    n_lines: int = 10000, 
    dt: float = 1.0, 
    dx: float = 1.0, 
    smoothing_radius: int = 1, 
    mode: str = 'linear'
):
    """
    Batch PyTorch implementation of Radon-like transform.
    Strictly runs on the device of the input tensor and returns GPU tensors.
    
    In 'circular' mode:
      1. Calculates circular center of mass for each image in batch.
      2. Shifts image to center this mass.
      3. Performs linear Radon transform on shifted image.
      4. Returns velocity and un-shifted intercept.
    """
    # 1. Setup & Dimensions
    # ------------------------------------------------------------------
    dev = images.device
    dtype = images.dtype
    
    is_single = False
    if images.ndim == 2:
        img = images.unsqueeze(0)
        is_single = True
    else:
        img = images
        
    batch_size, n_rows, n_cols = img.shape

    # 2. Pre-process (Batch Smoothing)
    # ------------------------------------------------------------------
    # Reshape for conv1d: (Batch * n_cols, 1, n_rows) -> Treat columns as independent channels temporarily
    # Actually, to smooth along rows (position), we permute to (Batch, Cols, Rows)
    img_permuted = img.permute(0, 2, 1).reshape(-1, 1, n_rows) # (B*T, 1, P)
    
    if smoothing_radius > 0:
        kernel_size = 2 * smoothing_radius + 1
        kernel = torch.ones((1, 1, kernel_size), device=dev, dtype=dtype)
        
        # Pad: Circular padding for circular mode is usually safer for smoothing
        pad_mode = 'circular' if mode == 'circular' else 'replicate'
        padded = F.pad(img_permuted, (smoothing_radius, smoothing_radius), mode=pad_mode)
        
        smoothed_flat = F.conv1d(padded, kernel)
    else:
        smoothed_flat = img_permuted

    # Reshape back to (Batch, n_rows, n_cols)
    smoothed = smoothed_flat.view(batch_size, n_cols, n_rows).permute(0, 2, 1)

    # 3. Handle Circular Shift Alignment
    # ------------------------------------------------------------------
    shift_amounts = torch.zeros(batch_size, device=dev, dtype=torch.long)
    
    if mode == 'circular':
        # A. Calculate Bin Centers
        y_edges = torch.linspace(0, 2 * np.pi, steps=n_rows + 1, device=dev, dtype=dtype)
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
        y_rad = y_centers.view(1, n_rows, 1) # (1, P, 1)

        # B. Calculate Global Circular Mean Phase per batch item
        sum_sin = torch.nansum(smoothed * torch.sin(y_rad), dim=(1, 2))
        sum_cos = torch.nansum(smoothed * torch.cos(y_rad), dim=(1, 2))
        
        mean_phase = torch.atan2(sum_sin, sum_cos)
        # Map to index [0, n_rows)
        mean_idx = ((mean_phase + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi) * n_rows
        
        # C. Calculate Shift
        center_target = n_rows // 2
        shift_amounts = torch.round(center_target - mean_idx).long()
        
        # D. Batch-wise Roll (Gather)
        # Construct grid indices
        y_base = torch.arange(n_rows, device=dev).view(1, n_rows, 1)
        shift_view = shift_amounts.view(batch_size, 1, 1)
        
        # Source index: (y - shift) % n_rows
        gather_indices_y = (y_base - shift_view) % n_rows
        gather_indices = gather_indices_y.expand(batch_size, n_rows, n_cols)
        
        working_image = torch.gather(smoothed, 1, gather_indices)
    else:
        working_image = smoothed

    # 4. Generate Random Trajectories
    # ------------------------------------------------------------------
    one_tensor = torch.tensor(1.0, device=dev, dtype=dtype)
    n_rows_tensor = torch.tensor(n_rows, device=dev, dtype=dtype)
    
    # Threshold to avoid vertical lines
    min_phi = torch.atan2(one_tensor, n_rows_tensor)
    
    # Generate Phi
    phi_candidates = (torch.rand(int(n_lines * 1.5), device=dev) * np.pi) - (np.pi / 2)
    valid_mask = torch.abs(phi_candidates) >= min_phi
    phi = phi_candidates[valid_mask]
    
    # Fallback if filtered too many
    if phi.numel() < n_lines:
        needed = n_lines - phi.numel()
        extra = (torch.rand(needed * 2, device=dev) * np.pi) - (np.pi / 2)
        phi = torch.cat([phi, extra[torch.abs(extra) >= min_phi]])
    phi = phi[:n_lines]

    # Generate Rho
    diag_sq = (n_cols - 1)**2 + (n_rows - 1)**2
    diag_len = torch.sqrt(torch.tensor(diag_sq, device=dev, dtype=dtype))
    rho = (torch.rand(n_lines, device=dev) * diag_len) - (diag_len / 2)

    # 5. Compute Line Coordinates & Extract Values
    # ------------------------------------------------------------------
    # Using 'working_image' which is now centered if circular, 
    # so we treat everything as LINEAR extraction to preserve slope sign.
    
    t_mid = (n_cols - 1) / 2
    y_mid = (n_rows - 1) / 2
    
    t_indices = torch.arange(n_cols, device=dev, dtype=dtype)
    
    sin_phi = torch.sin(phi).unsqueeze(1) # (n_lines, 1)
    cos_phi = torch.cos(phi).unsqueeze(1)
    rho_bc = rho.unsqueeze(1)

    # Line equation: y for each t
    # Shape: (n_lines, n_cols)
    y_lines_float = (rho_bc - (t_indices - t_mid) * cos_phi) / sin_phi + y_mid
    y_lines = torch.round(y_lines_float).long()

    # Expand for batch: (Batch, n_lines, n_cols)
    y_indices = y_lines.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Linear Extraction Strategy (for both linear and shifted-circular)
    # Clamp indices to valid range for gathering
    y_clamped = y_indices.clamp(0, n_rows - 1)
    
    # We need to gather from (Batch, n_rows, n_cols)
    # y_clamped is (Batch, n_lines, n_cols)
    # working_image is (Batch, n_rows, n_cols)
    # We need to expand working_image to allow gathering along dim 1 with n_lines
    # But gather requires index and input to have same number of dimensions.
    
    # Strategy: Use advanced indexing or gather.
    # Gather requires index dimension to match. 
    # Let's use F.grid_sample or manual gather. Manual gather is safer for exact indices.
    
    # Flatten batch and time for extraction? No, let's keep it clean.
    # We want result: (Batch, n_lines, n_cols)
    
    # To use torch.gather(input, dim, index), index must have same dims as input
    # Here input is (Batch, ROWS, Cols), Index is (Batch, LINES, Cols). 
    # Regular gather won't work directly because dimension size differs (ROWS vs LINES).
    # We must use advanced indexing.
    
    b_idx = torch.arange(batch_size, device=dev).view(batch_size, 1, 1)
    t_idx = torch.arange(n_cols, device=dev).view(1, 1, n_cols)
    
    # Advanced indexing: image[b, y, t]
    gathered_values = working_image[b_idx, y_clamped, t_idx] # (Batch, n_lines, n_cols)
    
    # Handle Out-of-Bounds (Median Fill)
    col_medians = torch.nanmedian(working_image, dim=1).values # (Batch, n_cols)
    medians_expanded = col_medians.unsqueeze(1).expand(-1, n_lines, -1)
    
    valid_mask = (y_indices >= 0) & (y_indices < n_rows)
    posterior = torch.where(valid_mask, gathered_values, medians_expanded)

    # 6. Find Best Line
    # ------------------------------------------------------------------
    line_scores = torch.nanmean(posterior, dim=2) # (Batch, n_lines)
    best_indices = torch.argmax(line_scores, dim=1) # (Batch,)
    
    batch_indices = torch.arange(batch_size, device=dev)
    best_scores = line_scores[batch_indices, best_indices]
    
    best_phi = phi[best_indices]
    best_rho = rho[best_indices]

    # 7. Convert to Physical Units & Correct Intercept
    # ------------------------------------------------------------------
    phys_t_mid = n_cols * dt / 2.0
    phys_p_mid = n_rows * dx / 2.0
    
    best_tan = torch.tan(best_phi)
    best_sin = torch.sin(best_phi)

    velocities = -dx / (dt * best_tan)
    
    raw_intercepts = (
        (dx * phys_t_mid) / (dt * best_tan) 
        + (best_rho / best_sin) * dx 
        + phys_p_mid
    )
    
    # Apply shift correction for circular mode
    # If we shifted the image UP by 5 pixels, the found intercept is 5 pixels higher 
    # than it would be in the original image. We subtract to get back to original.
    shift_phys = shift_amounts.to(dtype) * dx
    intercepts = raw_intercepts - shift_phys

    if is_single:
        return best_scores.squeeze(0), velocities.squeeze(0), intercepts.squeeze(0)

    return best_scores, velocities, intercepts


def wcorr(arr, mode='linear'):
    """
    Weighted correlation of a 2D array.
    
    Parameters
    ----------
    arr : np.ndarray
        Weighted 2D matrix (e.g., tuning curve). 
        Shape (ny, nx) -> (Phase, Position).
    mode : str
        'linear' : Standard Pearson correlation.
        'circular': Linear-Circular shift alignment method.
    """
    nx, ny = arr.shape[1], arr.shape[0]
    arr_sum = np.nansum(arr)
    
    if arr_sum == 0 or np.isnan(arr_sum):
        return np.nan
    
    # X axis is always linear (Position)
    x_mat = np.tile(np.arange(nx), (ny, 1))
    
    if mode == 'linear':
        y_mat = np.tile(np.arange(ny)[:, np.newaxis], (1, nx))
        
        ey = np.nansum(arr * y_mat) / arr_sum
        ex = np.nansum(arr * x_mat) / arr_sum
        
        cov_xy = np.nansum(arr * (y_mat - ey) * (x_mat - ex)) / arr_sum
        var_yy = np.nansum(arr * (y_mat - ey) ** 2) / arr_sum
        var_xx = np.nansum(arr * (x_mat - ex) ** 2) / arr_sum
        
        return cov_xy / np.sqrt(var_xx * var_yy)

    elif mode == 'circular':
        # --- CALCULATE BIN CENTERS USING ny+1 EDGES ---
        # 1. Create edges from 0 to 2*pi
        y_edges = np.linspace(0, 2 * np.pi, ny + 1, endpoint=True)
        # 2. Calculate the center of each bin (mean of adjacent edges)
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        y_rad = y_centers[:, np.newaxis]

        # --- CALCULATE GLOBAL CIRCULAR MEAN PHASE ---
        # Weight the sine and cosine components by the input array values
        sum_sin = np.nansum(arr * np.sin(y_rad))
        sum_cos = np.nansum(arr * np.cos(y_rad))
        
        # Determine the average phase angle
        mean_phase = np.arctan2(sum_sin, sum_cos)
        # Map phase back to index space [0, ny)
        # (mean_phase % 2pi) ensures we are in the [0, 2pi] range
        mean_idx = ((mean_phase + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi) * ny
        
        # --- CIRCULAR SHIFT TO ALIGN MEAN TO CENTER ---
        # Target the middle index of the Y axis to minimize edge effects
        center_target = ny // 2
        shift_amount = int(np.round(center_target - mean_idx))
        
        # Roll the array along the Y axis (axis=0)
        arr_shifted = np.roll(arr, shift=shift_amount, axis=0)
        
        # --- LINEAR CORRELATION ON SHIFTED DATA ---
        y_mat = np.tile(np.arange(ny)[:, np.newaxis], (1, nx))
        
        ey = np.nansum(arr_shifted * y_mat) / arr_sum
        ex = np.nansum(arr_shifted * x_mat) / arr_sum
        
        cov_xy = np.nansum(arr_shifted * (y_mat - ey) * (x_mat - ex)) / arr_sum
        var_yy = np.nansum(arr_shifted * (y_mat - ey) ** 2) / arr_sum
        var_xx = np.nansum(arr_shifted * (x_mat - ex) ** 2) / arr_sum
        
        return cov_xy / np.sqrt(var_xx * var_yy)

    else:
        raise ValueError("mode must be 'linear' or 'circular'")

@torch.inference_mode()
def wcorr_batch(arr: torch.Tensor, mode: str = 'linear'):
    """
    Batched Weighted correlation of 2D arrays (PyTorch / CUDA version).
    Strictly runs on the device of the input tensor and returns GPU tensors.
    
    Parameters
    ----------
    arr : torch.Tensor
        Weighted 2D matrices. 
        Shape: (Batch, ny, nx) -> (Batch, Phase, Position)
        OR (ny, nx) for single input.
    mode : str
        'linear' : Standard Pearson correlation.
        'circular': Calculates circular mean, shifts it to center, 
                   then calculates Pearson correlation (preserves sign).
        
    Returns
    -------
    torch.Tensor
        Correlation coefficients. Shape (Batch,) or scalar tensor if input was 2D.
    """
    # 1. Setup Device and Data
    # ------------------------------------------------------------------
    dev = arr.device
    dtype = arr.dtype
    
    # Handle single image input by adding batch dim
    is_single = False
    if arr.ndim == 2:
        batch_arr = arr.unsqueeze(0) 
        is_single = True
    else:
        batch_arr = arr
        
    batch_size, ny, nx = batch_arr.shape
    
    # Calculate Total Weight (Sum) per batch item
    arr_sum = torch.nansum(batch_arr, dim=(1, 2), keepdim=True)
    
    # Mask for zero-sum arrays
    valid_mask = (arr_sum.squeeze((1, 2)) > 0)
    
    # Safe sum for division (replace 0 with 1 to avoid nan/inf during intermediate steps)
    arr_sum_safe = arr_sum.clone()
    arr_sum_safe[arr_sum == 0] = 1.0

    # 2. Pre-processing: Handle Circular Shift if needed
    # ------------------------------------------------------------------
    if mode == 'circular':
        # --- A. CALCULATE BIN CENTERS ---
        # Create edges from 0 to 2*pi
        y_edges = torch.linspace(0, 2 * np.pi, steps=ny + 1, device=dev, dtype=dtype)
        # Calculate centers
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
        # Reshape for broadcasting: (1, ny, 1)
        y_rad = y_centers.view(1, ny, 1)

        # --- B. CALCULATE GLOBAL CIRCULAR MEAN PHASE ---
        # Weighted sum of sin and cos
        sum_sin = torch.nansum(batch_arr * torch.sin(y_rad), dim=(1, 2)) # (Batch,)
        sum_cos = torch.nansum(batch_arr * torch.cos(y_rad), dim=(1, 2)) # (Batch,)
        
        # Determine average phase angle
        mean_phase = torch.atan2(sum_sin, sum_cos)
        
        # Map phase back to index space [0, ny)
        # Note: (mean_phase + 2pi) % 2pi ensures positive range
        mean_idx = ((mean_phase + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi) * ny
        
        # --- C. BATCH-WISE CIRCULAR SHIFT ---
        center_target = ny // 2
        # Calculate shift amount: shift = target - current
        shift_amounts = torch.round(center_target - mean_idx).to(torch.long)
        
        # Vectorized Roll using torch.gather
        # 1. Create base Y indices: (1, ny, 1)
        y_base = torch.arange(ny, device=dev).view(1, ny, 1)
        
        # 2. Calculate source indices for each pixel in the shifted output
        #    Output[y] comes from Input[(y - shift) % ny]
        #    Shape: (Batch, ny, 1)
        shift_expanded = shift_amounts.view(batch_size, 1, 1)
        gather_indices_y = (y_base - shift_expanded) % ny
        
        # 3. Expand to match full tensor shape (Batch, ny, nx) for gather
        gather_indices = gather_indices_y.expand(batch_size, ny, nx)
        
        # 4. Perform the gather (effectively a batch-wise roll)
        use_arr = torch.gather(batch_arr, 1, gather_indices)
        
    else:
        # For linear mode, use original array
        use_arr = batch_arr

    # 3. Calculate Weighted Linear Correlation (On shifted or original data)
    # ------------------------------------------------------------------
    # Create Coordinate Grids (Shared across batch)
    y_idx = torch.arange(ny, device=dev, dtype=dtype)
    x_idx = torch.arange(nx, device=dev, dtype=dtype)
    
    # (ny, nx)
    y_mat, x_mat = torch.meshgrid(y_idx, x_idx, indexing='ij')
    
    # Expand to (Batch, ny, nx)
    y_mat = y_mat.unsqueeze(0).expand(batch_size, -1, -1)
    x_mat = x_mat.unsqueeze(0).expand(batch_size, -1, -1)

    # Weighted Means
    ey = torch.nansum(use_arr * y_mat, dim=(1, 2), keepdim=True) / arr_sum_safe
    ex = torch.nansum(use_arr * x_mat, dim=(1, 2), keepdim=True) / arr_sum_safe
    
    # Weighted Covariance and Variances
    y_diff = y_mat - ey
    x_diff = x_mat - ex
    
    denom_safe_sq = arr_sum_safe.squeeze((1, 2))

    cov_xy = torch.nansum(use_arr * y_diff * x_diff, dim=(1, 2)) / denom_safe_sq
    cov_yy = torch.nansum(use_arr * y_diff.square(), dim=(1, 2)) / denom_safe_sq
    cov_xx = torch.nansum(use_arr * x_diff.square(), dim=(1, 2)) / denom_safe_sq
    
    denominator = torch.sqrt(cov_xx * cov_yy)
    
    # Calculate R
    results = torch.zeros(batch_size, device=dev, dtype=dtype)
    
    # Combined mask: valid sum AND valid variance (avoid div by zero)
    valid_denom = (denominator > 0) & valid_mask
    
    results[valid_denom] = cov_xy[valid_denom] / denominator[valid_denom]
    
    if is_single:
        return results.squeeze(0) # Return scalar tensor
    return results


def jump_distance(posteriors, jump_stat="mean", mode="linear", norm=True):
    """
    Calculate jump distance for posterior matrices, ignoring invalid time bins.
    
    A time bin is considered "invalid" and excluded if:
    1. It contains all NaNs (e.g. no_spike_policy='nan')
    2. It is perfectly uniform/flat (e.g. no_spike_policy='uniform')
    
    Parameters
    ----------
    posteriors : list of np.ndarray
        List of posterior matrices (shape: n_bins x n_time_steps)
    jump_stat : str
        'mean', 'median', or 'max' aggregation of jumps.
    mode : str
        'linear': |x2 - x1|
        'circular': min(|x2 - x1|, N - |x2 - x1|)
    norm : bool
        If True, normalize by 1/n_bins (return 0 to 1).
    """

    if jump_stat == "mean":
        f = np.mean
    elif jump_stat == "median":
        f = np.median
    elif jump_stat == "max":
        f = np.max
    else:
        raise ValueError("Invalid jump_stat. Valid values: mean, median, max")

    if len(posteriors) == 0:
        return np.array([])

    # Number of bins
    n_bins = posteriors[0].shape[0]
    
    # Scale factor
    dx = 1.0 / n_bins if norm else 1.0

    results = []
    
    for p in posteriors:
        # --- 1. Identify Valid Columns ---
        # Check A: Is the column entirely NaN?
        is_all_nan = np.all(np.isnan(p), axis=0)
        
        # Check B: Is the column Uniform? (Max value approx equals Min value)
        # We use peak-to-peak (ptp) with a tiny epsilon for float safety
        is_uniform = np.ptp(p, axis=0) < 1e-9
        
        # A bin is valid if it is NEITHER NaN NOR Uniform
        valid_mask = ~(is_all_nan | is_uniform)
        
        # If we have fewer than 2 valid points, we cannot calculate a jump
        if np.sum(valid_mask) < 2:
            results.append(np.nan)
            continue

        # --- 2. Get Peaks safely ---
        # Fill NaNs with 0 temporarily so argmax doesn't crash on all-NaN columns.
        # (The values in invalid columns don't matter because we filter them out later)
        p_safe = p.copy()
        p_safe[np.isnan(p)] = 0
        peaks = np.argmax(p_safe, axis=0)
        
        # --- 3. Calculate Diffs ---
        raw_diffs = np.abs(np.diff(peaks))
        
        # Apply Circular Logic
        if mode == 'circular':
            raw_diffs = np.minimum(raw_diffs, n_bins - raw_diffs)

        # --- 4. Filter Invalid Jumps ---
        # A jump is valid only if BOTH time t and time t+1 were valid
        valid_transitions = valid_mask[:-1] & valid_mask[1:]
        
        if np.sum(valid_transitions) == 0:
            results.append(np.nan)
        else:
            # Only aggregate the diffs that happened between valid bins
            valid_diffs = raw_diffs[valid_transitions]
            results.append(f(valid_diffs))

    return np.array(results) * dx

def column_shift(arr, shifts=None):
    """Circular shift columns independently by a given amount"""

    assert arr.ndim == 2, "only 2d arrays accepted"

    if shifts is None:
        rng = np.random.default_rng()
        shifts = rng.integers(-arr.shape[0], arr.shape[0], arr.shape[1])

    assert arr.shape[1] == len(shifts)

    shifts = shifts % arr.shape[0]
    rows_indx, columns_indx = np.ogrid[: arr.shape[0], : arr.shape[1]]

    rows_indx = rows_indx - shifts[np.newaxis, :]

    return arr[rows_indx, columns_indx]