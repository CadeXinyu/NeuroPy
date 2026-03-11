import scipy.signal as sg
import numpy as np


def detect_epochs_peak(signal, edge_cutoff, lowthresh, highthresh, prominence=0):
    """
    Detect peaks and merge overlapping epochs.
    
    Parameters
    ----------
    signal : np.ndarray
        1D array of signal values (can contain NaN for excluded regions)
    edge_cutoff : float
        Threshold value for initial detection (values below this are set to -100)
    lowthresh : float
        Minimum peak height
    highthresh : float
        Maximum peak height
    prominence : float, default=0
        Minimum prominence of peaks (required to get left_bases and right_bases)
    
    Returns
    -------
    merged_epochs : np.ndarray
        Array of shape (n_epochs, 4) with columns [start_idx, stop_idx, peak_idx, peak_amp]
        Returns empty array if no peaks detected
    """
    signal = signal.copy()
    # Apply edge cutoff threshold (NaN values will remain NaN)
    amp_thresh = np.where(
        signal >= edge_cutoff, 
        signal, 
        np.nan
    )
    
    # NaN values remain as NaN - find_peaks treats them as natural boundaries
    # (No np.nan_to_num needed)
    
    # Find peaks within threshold range
    peaks, props = sg.find_peaks(
        amp_thresh, 
        height=[lowthresh, highthresh], 
        prominence=prominence
    )
    
    if len(peaks) == 0:
        return np.array([]).reshape(0, 4)
    
    starts = props["left_bases"]
    stops = props["right_bases"]
    peaks_amp = amp_thresh[peaks]
    
    # Merge overlapping epochs
    if len(peaks) == 1:
        # Only one peak, no merging needed
        merged_epochs = np.array([[starts[0], stops[0], peaks[0], peaks_amp[0]]])
    else:
        merged_epochs = []
        current_start = starts[0]
        current_stop = stops[0]
        current_peak = peaks[0]
        current_peak_amp = peaks_amp[0]
        
        for i in range(1, len(starts)):
            # Check if current epoch overlaps with previous
            if starts[i] <= current_stop:
                # Merge: extend boundaries and keep highest peak
                current_stop = max(current_stop, stops[i])
                if peaks_amp[i] > current_peak_amp:
                    current_peak = peaks[i]
                    current_peak_amp = peaks_amp[i]
            else:
                # Save previous epoch and start new one
                merged_epochs.append([current_start, current_stop, current_peak, current_peak_amp])
                current_start = starts[i]
                current_stop = stops[i]
                current_peak = peaks[i]
                current_peak_amp = peaks_amp[i]
        
        # Don't forget the last epoch
        merged_epochs.append([current_start, current_stop, current_peak, current_peak_amp])
        merged_epochs = np.array(merged_epochs)
    
    return merged_epochs


def detect_epochs_thres(signal, thresh, edge_cutoff=None):
    """
    Detect epochs where signal exceeds threshold using binary detection.
    
    Parameters
    ----------
    signal : np.ndarray
        1D array of signal values (can contain NaN for excluded regions)
    thresh : float
        Primary threshold for detection
    edge_cutoff : float, optional
        Lower threshold for extending epoch boundaries. If None, no extension is performed.
    
    Returns
    -------
    epochs : np.ndarray
        Array of shape (n_epochs, 2) with columns [start_idx, end_idx]
        Returns empty array if no epochs detected
    """
    # Work on a copy to avoid modifying original signal
    signal = signal.copy()
    
    # Stage 1: Find epochs above primary threshold
    # Create binary mask: 1 where signal exceeds threshold
    # NaN > thresh returns False, so NaN values will be 0 in the mask
    artifact_binary = (signal > thresh).astype(int)
    
    # Add padding for edge detection
    artifact_binary = np.concatenate(([0], artifact_binary, [0]))
    
    # Find transitions using diff: 0→1 (start) and 1→0 (end)
    artifact_diff = np.diff(artifact_binary)
    artifact_start = np.where(artifact_diff == 1)[0]
    artifact_end = np.where(artifact_diff == -1)[0]
    
    # Check if any epochs were found
    if len(artifact_start) == 0:
        return np.array([]).reshape(0, 2)
    
    # Combine starts and ends into array of [start, end] pairs
    epochs = np.column_stack((artifact_start, artifact_end))
    
    # Stage 2: Extend epoch boundaries to edge_cutoff threshold (optional)
    if edge_cutoff is not None:
        # Create binary mask at lower threshold for boundary extension
        edge_binary = (signal > edge_cutoff).astype(int)
        edge_binary = np.concatenate(([0], edge_binary, [0]))
        
        # Find all transitions at edge_cutoff threshold
        edge_diff = np.diff(edge_binary)
        edge_start = np.where(edge_diff == 1)[0]
        edge_end = np.where(edge_diff == -1)[0]
        
        # Extend each epoch's boundaries
        extended_epochs = []
        for start, end in epochs:
            # Find the nearest edge_start that is <= start
            valid_edge_starts = edge_start[edge_start <= start]
            new_start = valid_edge_starts[-1] if len(valid_edge_starts) > 0 else start
            
            # Find the nearest edge_end that is >= end
            valid_edge_ends = edge_end[edge_end >= end]
            new_end = valid_edge_ends[0] if len(valid_edge_ends) > 0 else end
            
            extended_epochs.append([new_start, new_end])
        
        epochs = np.array(extended_epochs)
    
    return epochs