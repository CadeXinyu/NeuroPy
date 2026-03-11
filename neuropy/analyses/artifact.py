import pandas as pd
import numpy as np
from ..core import Signal
from neuropy.utils.detect import detect_epochs_thres
import scipy.stats as stats
from ..core import Epoch


def detect_artifact_epochs(
    signal: Signal,
    thresh: float = 4,
    edge_cutoff: float = 2,
    filt: list or tuple = None,
    verbose: bool = False
):
    """
    Detect artifact periods using z-score measure with optional Hilbert envelope.
    
    Identifies noisy epochs in neural signals by computing z-scores of the
    averaged signal across channels and detecting periods that exceed specified
    thresholds. When filtering is applied, uses Hilbert envelope to detect
    amplitude-based artifacts. Extends boundaries to edge cutoff threshold.
    
    Parameters
    ----------
    signal : Signal
        Signal object containing neural data
    thresh : float, optional
        Z-score threshold above which signal is considered artifactual.
        Default is 4 (4 standard deviations from mean)
    edge_cutoff : float, optional
        Z-score value to which artifact boundaries are extended.
        Must be <= thresh. Default is 2
    filt : list or tuple, optional
        Frequency band for filtering [low_freq, high_freq] in Hz
        When provided, applies bandpass filter followed by Hilbert envelope
        - [3, 3000]: bandpass between 3-3000 Hz
        - [45, None]: highpass above 45 Hz  
        - [None, 200]: lowpass below 200 Hz
        - None: no filtering (default)
    verbose : bool, optional
        Whether to show MNE filtering progress and artifact summary. 
        Default is False
        
    Returns
    -------
    Epoch or None
        Epoch object containing artifact periods with start/stop times.
        Returns None if no artifacts found
        
    Raises
    ------
    AssertionError
        If edge_cutoff > thresh or if filt is not length 2
        
    Notes
    -----
    When filtering is applied, the function uses Hilbert envelope to detect
    amplitude-based artifacts in the specified frequency band. This is useful
    for detecting artifacts in specific frequency ranges (e.g., muscle artifacts
    in high frequencies).
    
    Examples
    --------
    >>> # Detect artifacts with 4 sigma threshold (no filtering)
    >>> artifacts = detect_artifact_epochs(signal, thresh=4)
    >>> 
    >>> # Detect high-frequency artifacts (muscle/EMG) with bandpass + Hilbert
    >>> artifacts = detect_artifact_epochs(signal, thresh=3, filt=[70, 200])
    >>>
    >>> # Detect movement artifacts in low frequencies
    >>> artifacts = detect_artifact_epochs(signal, thresh=3.5, filt=[0.5, 10])
    """
    
    # Create a copy to avoid modifying the original signal
    signal_work = signal.copy()
    
    # Validate parameters
    if edge_cutoff > thresh:
        raise ValueError(f"edge_cutoff ({edge_cutoff}) cannot exceed thresh ({thresh})")
    
    # Store sampling rate for time conversions
    sampling_rate = signal_work.sampling_rate
    
    # Apply filtering and Hilbert envelope if specified
    use_hilbert = False
    if filt is not None:
        if len(filt) != 2:
            raise ValueError(f"Filter specification must be [low_freq, high_freq], got {filt}")
        
        # Apply bandpass filter followed by Hilbert envelope for amplitude detection
        # This is particularly useful for detecting artifacts in specific frequency bands
        if verbose:
            print(f"Applying bandpass filter [{filt[0]}, {filt[1]}] Hz with Hilbert envelope...")
        
        signal_work.mne.filter(
            l_freq=filt[0], 
            h_freq=filt[1], 
            verbose=verbose
        )
        
        # Apply Hilbert transform to get amplitude envelope
        # This converts oscillations to smooth amplitude measure
        signal_work.mne.apply_hilbert(envelope=True)
        
        # Synchronize traces with processed MNE data
        signal_work.update()
        use_hilbert = True
    
    # Get processed traces
    traces = signal_work.traces
    
    # Average across all channels to get single representative trace
    # This reduces channel-specific noise and creates unified artifact detection
    trace = np.mean(traces, axis=0)
    
    # ---- Stage 1: Z-score computation and initial artifact detection ----
    # Compute z-scores (no absolute value - detecting only positive deviations)
    # This is appropriate for Hilbert envelope which is always positive
    if use_hilbert:
        zsc = stats.zscore(trace, axis=-1, nan_policy='omit')
    else:
        zsc = np.abs(stats.zscore(trace, axis=-1, nan_policy='omit'))
    
    # Handle edge case where signal is constant (std = 0)
    if np.all(np.isnan(zsc)):
        if verbose:
            print("Signal has zero variance, cannot compute z-scores")
        return None
    
    # Detect epochs using binary threshold detection
    artifact_epochs = detect_epochs_thres(signal=zsc, thresh=thresh, edge_cutoff=edge_cutoff)
    
    # Check if any artifacts were found
    if len(artifact_epochs) == 0:
        if verbose:
            thresh_type = "amplitude" if use_hilbert else "z-score"
            print(f"No artifacts found at {thresh_type} threshold {thresh}")
        return None
    
    # ---- Stage 2: Convert to time and create output ----
    # Convert from samples to seconds, accounting for signal's start time
    artifact_times = artifact_epochs / sampling_rate + signal.t_start
    
    # Create DataFrame with artifact periods
    epochs_df = pd.DataFrame({
        "start": artifact_times[:, 0],
        "stop": artifact_times[:, 1],
        "label": "artifact"
    })
    
    # Calculate comprehensive statistics
    durations = artifact_times[:, 1] - artifact_times[:, 0]
    total_artifact_duration = np.sum(durations)
    artifact_percentage = (total_artifact_duration / signal.duration) * 100
    
    # Create detailed metadata
    metadata = {
        "threshold": thresh,
        "edge_cutoff": edge_cutoff,
        "filter_applied": filt,
        "hilbert_envelope": use_hilbert,
        "n_artifacts": len(epochs_df),
        "total_artifact_sec": float(total_artifact_duration),
        "artifact_percentage": float(artifact_percentage),
        "mean_duration_sec": float(np.mean(durations)),
        "std_duration_sec": float(np.std(durations)),
        "min_duration_sec": float(np.min(durations)),
        "max_duration_sec": float(np.max(durations)),
        "sampling_rate": sampling_rate
    }
    
    # Add source file information if available
    if hasattr(signal, 'source_file') and signal.source_file is not None:
        from pathlib import Path
        metadata["source_file"] = str(Path(signal.source_file))
    
    # Create Epoch object with artifacts
    art_epochs = Epoch(epochs_df, metadata=metadata)
    
    # Print detailed summary if verbose
    if verbose:
        print(f"\nArtifact Detection Summary:")
        print(f"  Found {len(epochs_df)} artifact epochs")
        print(f"  Total duration: {total_artifact_duration:.2f} sec ({artifact_percentage:.1f}% of signal)")
        print(f"  Mean duration: {np.mean(durations):.2f} ± {np.std(durations):.2f} sec")
        print(f"  Range: [{np.min(durations):.2f}, {np.max(durations):.2f}] sec")
        if use_hilbert:
            print(f"  Method: Hilbert envelope in band [{filt[0]}-{filt[1]}] Hz")
    
    return art_epochs


if __name__ == "__main__":
    print("test")