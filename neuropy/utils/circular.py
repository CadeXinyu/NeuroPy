import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d


def smooth_2d_circular_position(arr, sigma_pos, sigma_time=None, truncate=4.0):
    """
    Smooth 2D array with circular boundary on position axis, linear on time axis.
    
    Parameters
    ----------
    arr : np.ndarray, shape (n_pos_bins, n_time_bins)
        2D array to smooth (e.g., posterior probability)
    sigma_pos : float
        Smoothing sigma for position axis (axis=0) in bin units.
        Uses circular boundary conditions.
    sigma_time : float, optional
        Smoothing sigma for time axis (axis=1) in bin units.
        Uses standard boundary (reflect). If None, no time smoothing.
    truncate : float, default 4.0
        Truncate filter at this many sigmas
        
    Returns
    -------
    np.ndarray
        Smoothed array, same shape as input
    """
    arr = np.asarray(arr, dtype=float)
    
    if sigma_pos <= 0 and (sigma_time is None or sigma_time <= 0):
        return arr.copy()
    
    smoothed = arr.copy()
    
    # Circular smoothing along position axis (axis=0)
    if sigma_pos > 0:
        n_pos, n_time = smoothed.shape
        pad_width = int(np.ceil(truncate * sigma_pos))
        
        if pad_width >= n_pos:
            # Very large sigma: tile entire array
            padded = np.concatenate([smoothed, smoothed, smoothed], axis=0)
            smoothed_padded = gaussian_filter1d(padded, sigma=sigma_pos, axis=0, truncate=truncate)
            smoothed = smoothed_padded[n_pos:2*n_pos, :]
        else:
            # Circular pad along axis=0
            padded = np.concatenate([
                smoothed[-pad_width:, :],  # bottom rows to top
                smoothed,
                smoothed[:pad_width, :]    # top rows to bottom
            ], axis=0)
            smoothed_padded = gaussian_filter1d(padded, sigma=sigma_pos, axis=0, truncate=truncate)
            smoothed = smoothed_padded[pad_width:-pad_width, :]
    
    # Linear smoothing along time axis (axis=1)
    if sigma_time is not None and sigma_time > 0:
        smoothed = gaussian_filter1d(smoothed, sigma=sigma_time, axis=1, truncate=truncate)
    
    return smoothed

def interpolate_phase_circular(time, phase, query_times):
    """
    Interpolate circular phase data at arbitrary query times using PCHIP interpolation.
    
    Converts phase to x,y coordinates, interpolates, then converts back.
    Preserves input array structure.
    
    UPGRADE: 
    1. Ignores NaN values in inputs.
    2. RAISES ERROR if interpolation fails or not enough data (no silent return).
    
    Parameters
    ----------
    time : array-like
        Time points of phase measurements
    phase : array-like
        Phase values in radians (can contain NaNs)
    query_times : array-like
        Times at which to interpolate phase (any shape)
        
    Returns
    -------
    interpolated_phase : ndarray
        Phase values at query times.
    
    Raises
    ------
    ValueError
        If there are fewer than 2 valid data points after removing NaNs.
    ValueError
        If PchipInterpolator fails (e.g., time points are not strictly increasing).
    """
    # Ensure inputs are numpy arrays
    time = np.asarray(time)
    phase = np.asarray(phase)
    
    # Store original shape
    original_shape = np.asarray(query_times).shape
    query_times_flat = np.asarray(query_times).flatten()
    
    # --- STEP 1: Handle NaNs ---
    # Create a mask where phase AND time are valid
    valid_mask = ~np.isnan(phase) & ~np.isnan(time)
    
    # Apply mask to filter out invalid data
    clean_time = time[valid_mask]
    clean_phase = phase[valid_mask]
    
    # --- STEP 2: Check for Data Sufficiency ---
    if len(clean_time) < 2:
        raise ValueError(f"Interpolation failed: Not enough valid data points. "
                         f"Input had {len(time)} points, but only {len(clean_time)} were valid (non-NaN).")
    
    # --- STEP 3: Interpolation ---
    # Convert clean phase to x,y coordinates
    cos_vals = np.cos(clean_phase)
    sin_vals = np.sin(clean_phase)
    
    # We let PchipInterpolator raise its own ValueError if it fails
    # (e.g., if clean_time is not strictly increasing)
    f_cos = PchipInterpolator(clean_time, cos_vals, extrapolate=True)
    f_sin = PchipInterpolator(clean_time, sin_vals, extrapolate=True)
    
    # Get interpolated x,y at query times
    query_cos = f_cos(query_times_flat)
    query_sin = f_sin(query_times_flat)
    
    # Convert back to phase
    interpolated_phase = np.arctan2(query_sin, query_cos)
    interpolated_phase = np.mod(interpolated_phase, 2 * np.pi)
    
    # Reshape to original shape
    return interpolated_phase.reshape(original_shape)


def circular_smooth(counts, sigma_bins):
    """
    Smooth histogram counts with circular boundary conditions.
    
    Parameters
    ----------
    counts : array-like
        Histogram counts
    sigma_bins : float
        Smoothing sigma in bin units
        
    Returns
    -------
    smoothed_counts : ndarray
        Smoothed counts
    """
    # Pad with wrapping for circular boundary
    pad_size = int(3 * sigma_bins)  # Pad 3 sigma on each side
    counts_padded = np.concatenate([counts[-pad_size:], counts, counts[:pad_size]])
    
    # Smooth
    counts_smoothed = gaussian_filter1d(counts_padded.astype(float), sigma=sigma_bins, mode='wrap')
    
    # Extract center
    return counts_smoothed[pad_size:-pad_size]

import numpy as np
from scipy.optimize import brute, fmin
from scipy.special import erf

def circular_linear_regression(x, y_circular, slope_bounds=None):
    """
    Performs circular-linear regression of circular data y_circular on linear data x
    using the method described in Kempter et al. (2012).
    
    MODIFIED: 
    - Normalizes x to [0, 1] internally.
    - Returns slope in radians per normalized unit (radians per field width).
    - Returns intercept at x_min (which is x_norm = 0).
    
    Parameters:
    -----------
    x : array-like
        Linear variable (e.g., position).
    y_circular : array-like
        Circular variable in RADIANS (e.g., spike phase).
    slope_bounds : tuple (min, max), optional
        Search range for the slope 'a' (in cycles per FIELD WIDTH). 
        If None, defaults to +/- 5 cycles.
        
    Returns:
    --------
    dict containing regression metrics. Returns p=1 and NaNs if n < 2.
    """
    # Convert to numpy arrays
    x = np.array(x)
    phi = np.array(y_circular)
    
    # Filter out NaNs to ensure we are counting valid data points
    valid_mask = ~np.isnan(x) & ~np.isnan(phi)
    x = x[valid_mask]
    phi = phi[valid_mask]
    
    n = len(x)
    
    if n < 2:
        return {
            'slope': np.nan,
            'intercept': np.nan,
            'r': np.nan,
            'signed_r': np.nan,
            'r_squared': np.nan,
            'pvalue': 1.0
        }

    # --- 1. Normalize x to [0, 1] ---
    x_min = np.min(x)
    x_max = np.max(x)
    range_x = x_max - x_min
    
    if range_x > 0:
        x_norm = (x - x_min) / range_x
    else:
        x_norm = np.zeros_like(x)

    # -------------------------------------------------------------------------
    # 2. Define Mean Resultant Length R(a)
    # R(a) = sqrt( [sum cos(phi - 2pi*a*x_norm)]^2 + [sum sin(phi - 2pi*a*x_norm)]^2 ) / n
    # -------------------------------------------------------------------------
    def get_neg_R(a):
        # a is in cycles per NORMALIZED unit (i.e., cycles per field)
        theta_pred = 2 * np.pi * a * x_norm
        resid = phi - theta_pred
        
        C = np.sum(np.cos(resid)) / n
        S = np.sum(np.sin(resid)) / n
        R = np.sqrt(C**2 + S**2)
        return -R

    # -------------------------------------------------------------------------
    # 3. Estimate Slope (a)
    # -------------------------------------------------------------------------
    if slope_bounds is None:
        # Default: search +/- 5 cycles over the entire field width
        slope_bounds = (-1.5, 1.5)

    # Coarse grid search
    grid_res = brute(get_neg_R, ranges=(slope_bounds,), Ns=100, full_output=True, finish=None)
    best_a_guess = grid_res[0]
    
    # Fine-tune using local minimization
    best_a = fmin(get_neg_R, best_a_guess, disp=False)[0]
    
    # Calculate the observed R at this optimal slope
    observed_R = -get_neg_R(best_a)
    
    # Convert 'a' (cycles/field) to slope in radians/field
    # NO conversion back to real units requested.
    slope_rads_norm = 2 * np.pi * best_a

    # -------------------------------------------------------------------------
    # 4. Estimate Intercept (phi_0)
    # Because x is normalized, x_norm=0 corresponds to x=x_min.
    # So phi_0 IS the intercept at the start of the field.
    # -------------------------------------------------------------------------
    resid = phi - slope_rads_norm * x_norm
    sum_sin = np.sum(np.sin(resid))
    sum_cos = np.sum(np.cos(resid))
    phi_0 = np.arctan2(sum_sin, sum_cos) 
    
    # -------------------------------------------------------------------------
    # 5. Calculate Circular-Linear Correlation (rho_c)
    # Transform normalized x to circular variable theta = |slope_norm| * x_norm
    # -------------------------------------------------------------------------
    theta = (np.abs(slope_rads_norm) * x_norm) % (2 * np.pi)
    
    def circ_mean(angles):
        return np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))

    mean_phi = circ_mean(phi)
    mean_theta = circ_mean(theta)
    
    sin_phi_diff = np.sin(phi - mean_phi)
    sin_theta_diff = np.sin(theta - mean_theta)
    
    numerator = np.sum(sin_phi_diff * sin_theta_diff)
    denominator = np.sqrt(np.sum(sin_phi_diff**2) * np.sum(sin_theta_diff**2))
    
    if denominator == 0:
        rho_c = 0.0
    else:
        rho_c = numerator / denominator

    # -------------------------------------------------------------------------
    # 6. Calculate P-value (Analytic Z-test)
    # -------------------------------------------------------------------------
    lambda_20 = np.mean(sin_phi_diff**2)
    lambda_02 = np.mean(sin_theta_diff**2)
    lambda_22 = np.mean((sin_phi_diff**2) * (sin_theta_diff**2))
    
    if lambda_22 < 1e-9:
        z = 0.0
    else:
        z = rho_c * np.sqrt((n * lambda_20 * lambda_02) / lambda_22)
    
    pvalue = 1.0 - erf(np.abs(z) / np.sqrt(2.0))

    return {
        'slope': slope_rads_norm,     # Radians per FIELD WIDTH (normalized)
        'intercept': phi_0,           # Phase at x_min
        'r': observed_R,              # Mean Resultant Length
        'signed_r': rho_c,            # Correlation
        'r_squared': rho_c**2,
        'pvalue': pvalue,
    }

import numpy as np
import torch
from scipy.special import erf

def circular_linear_regression_cuda(x, y_circular, n_permutations=1000, slope_range=(-1.5, 1.5), n_search=100, device='cuda'):
    """
    Optimized Circular-Linear Regression (Kempter et al., 2012) - CUDA.
    
    Parameters:
    -----------
    x : array-like
        Linear variable (e.g., position).
    y_circular : array-like
        Circular variable in RADIANS.
    slope_range : tuple
        Search limits in CYCLES per FIELD WIDTH. 
        Default (-1.5, 1.5) is biologically plausible for one place field.
    
    Returns:
    --------
    dict containing regression metrics (slope, intercept, r, signed_r, p-values).
    """
    
    # --- 1. Data Prep ---
    x = np.asarray(x)
    y_circular = np.asarray(y_circular)
    
    # Remove NaN
    valid_mask = ~(np.isnan(x) | np.isnan(y_circular))
    x = x[valid_mask]
    y_circular = y_circular[valid_mask]
    
    if len(x) < 2:
        return {
            'slope': np.nan, 'intercept': np.nan, 
            'r': np.nan, 'signed_r': np.nan, 'r_squared': np.nan,
            'pvalue_analytic': 1.0, 'pvalue_permutation': 1.0, 
            'predicted': np.full_like(x, np.nan)
        }
    
    # Device management
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)
    
    x_gpu = torch.tensor(x, dtype=torch.float32, device=device)
    y_gpu = torch.tensor(y_circular, dtype=torch.float32, device=device)
    
    # Normalize x to [0, 1]
    x_min = x_gpu.min()
    x_max = x_gpu.max()
    range_x = x_max - x_min
    
    if range_x > 1e-6:
        x_norm_gpu = (x_gpu - x_min) / range_x
    else:
        x_norm_gpu = torch.zeros_like(x_gpu)
    
    n_points = len(x_norm_gpu)

    # --- Helpers ---
    def get_analytical_intercept(x_vals, y_vals, slopes):
        if slopes.ndim == 0: slopes = slopes.unsqueeze(0)
        if y_vals.ndim == 1: y_vals = y_vals.unsqueeze(0)
            
        resids = y_vals - slopes[:, None] * x_vals[None, :]
        sum_sin = torch.sum(torch.sin(resids), dim=1)
        sum_cos = torch.sum(torch.cos(resids), dim=1)
        return torch.atan2(sum_sin, sum_cos)

    def get_rho_c(x_vals, y_vals, slopes):
        if slopes.ndim == 0: slopes = slopes.unsqueeze(0)
        if y_vals.ndim == 1: y_vals = y_vals.unsqueeze(0)
        
        x_in = x_vals.unsqueeze(0) 

        # x -> theta = |slope| * x
        theta = torch.abs(slopes)[:, None] * x_in 
        
        # Circular Means
        def circ_mean(angles):
            s = torch.sum(torch.sin(angles), dim=1, keepdim=True)
            c = torch.sum(torch.cos(angles), dim=1, keepdim=True)
            return torch.atan2(s, c)
            
        mean_phi = circ_mean(y_vals)
        mean_theta = circ_mean(theta)
        
        # Covariance
        sin_phi_diff = torch.sin(y_vals - mean_phi)
        sin_theta_diff = torch.sin(theta - mean_theta)
        
        numerator = torch.sum(sin_phi_diff * sin_theta_diff, dim=1)
        
        sum_sq_phi = torch.sum(sin_phi_diff**2, dim=1)
        sum_sq_theta = torch.sum(sin_theta_diff**2, dim=1)
        denominator = torch.sqrt(sum_sq_phi * sum_sq_theta)
        
        rho_c = torch.zeros_like(numerator)
        mask = denominator > 1e-6
        rho_c[mask] = numerator[mask] / denominator[mask]
        
        return rho_c, sin_phi_diff, sin_theta_diff

    # --- 2. Grid Search (Matrix Multiplication Method) ---
    # Convert cycles to radians for the search grid (2*pi*a)
    cycles_min, cycles_max = slope_range
    search_slopes = torch.linspace(cycles_min * 2*np.pi, cycles_max * 2*np.pi, n_search, device=device)
    
    # Pre-calculate Complex X term: exp(-i * slope * x_norm)
    # Shape: (n_points, n_search)
    term_x = torch.exp(-1j * search_slopes[None, :] * x_norm_gpu[:, None])
    
    # 2a. Observed Grid Search
    term_y_obs = torch.exp(1j * y_gpu).unsqueeze(0) 
    Z_grid_obs = torch.matmul(term_y_obs, term_x)
    R_grid_obs = torch.abs(Z_grid_obs).squeeze(0)
    
    best_idx = torch.argmax(R_grid_obs)
    initial_slope = search_slopes[best_idx]
    
    # --- 3. Fine-Tune Slope (L-BFGS) ---
    slope_param = initial_slope.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS([slope_param], max_iter=50, line_search_fn='strong_wolfe')
    
    def closure():
        optimizer.zero_grad()
        resids = y_gpu - slope_param * x_norm_gpu
        Z = torch.mean(torch.exp(1j * resids))
        loss = -torch.abs(Z)
        loss.backward()
        return loss
        
    optimizer.step(closure)
    best_slope_norm = slope_param.detach() # Rads per field width

    # --- 4. Final Metrics (Observed) ---
    
    # A. Intercept (Phase at x_min)
    intercept_norm = get_analytical_intercept(x_norm_gpu, y_gpu, best_slope_norm)[0]
    
    # B. Mean Resultant Length (R)
    # We calculate R explicitly for the best slope
    resids_final = y_gpu - best_slope_norm * x_norm_gpu
    R_obs = torch.abs(torch.mean(torch.exp(1j * resids_final)))
    
    # C. Correlation (rho_c)
    rho_c_obs, sin_diff_phi, sin_diff_theta = get_rho_c(x_norm_gpu, y_gpu, best_slope_norm)
    rho_c_obs = rho_c_obs[0]
    
    # --- 5. Analytical P-Value ---
    lambda_20 = torch.mean(sin_diff_phi**2)
    lambda_02 = torch.mean(sin_diff_theta**2)
    lambda_22 = torch.mean((sin_diff_phi**2) * (sin_diff_theta**2))
    
    if lambda_22 > 1e-9:
        z_stat = rho_c_obs * torch.sqrt((n_points * lambda_20 * lambda_02) / lambda_22)
        p_val_analytic = 1.0 - erf(torch.abs(z_stat.cpu()).numpy() / np.sqrt(2.0))
    else:
        p_val_analytic = 1.0

    # --- 6. Permutation Test (Matrix Mult Optimized) ---
    p_val_perm = np.nan
    if n_permutations > 0:
        # Shuffle Y indices
        perm_indices = torch.stack([torch.randperm(n_points, device=device) for _ in range(n_permutations)])
        y_perms = y_gpu[perm_indices]
        
        # Matrix Mult Grid Search
        term_y_perms = torch.exp(1j * y_perms)
        Z_grid_perms = torch.matmul(term_y_perms, term_x)
        R_grid_perms = torch.abs(Z_grid_perms)
        
        # Best slope for each perm
        best_indices_perm = torch.argmax(R_grid_perms, dim=1)
        best_slopes_perm = search_slopes[best_indices_perm]
        
        # Rho_c for each perm
        rho_c_perms, _, _ = get_rho_c(x_norm_gpu, y_perms, best_slopes_perm)
        
        count = torch.sum(torch.abs(rho_c_perms) >= torch.abs(rho_c_obs))
        p_val_perm = count.item() / n_permutations

    # --- 7. Output ---
    slope_val = best_slope_norm.item()
    intercept_val = intercept_norm.item()
    
    # Prediction uses normalized variables
    y_pred = torch.remainder(slope_val * x_norm_gpu + intercept_val, 2*np.pi)
    
    return {
        'slope': float(slope_val),        # Radians per FIELD WIDTH
        'intercept': float(intercept_val),# Phase at x_min
        'r': float(R_obs.item()),         # Mean Resultant Length (Vector Strength)
        'signed_r': float(rho_c_obs.item()), # Circular-Linear Correlation
        'r_squared': float(rho_c_obs.item()**2),
        'pvalue_analytic': float(p_val_analytic),
        'pvalue_permutation': float(p_val_perm), # Usually designated as 'pvalue' in your CPU dict
        'pvalue': float(p_val_perm),             # Alias for compatibility
        'predicted': y_pred.cpu().numpy()
    }