import numpy as np
import pandas as pd
from neuropy.utils import mathutil, signal_process
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import scipy.signal as sg
from copy import deepcopy

from neuropy.core import Signal, ProbeGroup, Epoch
from neuropy.utils.signal_process import WaveletSg, filter_sig
from neuropy.io import BinarysignalIO
from neuropy.utils.detect import detect_epochs_peak


def _detect_freq_band_epochs(
    signals,
    freq_band,
    thresh,
    edge_cutoff,
    mindur,
    maxdur,
    mergedist,
    fs,
    sigma,
    ignore_times=None,
    return_power=False,
):
    """Detects epochs of high power in a given frequency band

    Parameters
    ----------
    thresh : tuple, optional
        low and high threshold for detection
    mindur : float, optional
        minimum duration of epoch
    maxdur : float, optional
    chans : list
        channels used for epoch detection, if None then chooses best chans
    """

    lf, hf = freq_band
    dt = 1 / fs
    smooth = lambda x: gaussian_filter1d(x, sigma=sigma / dt, axis=-1)
    lowthresh, highthresh = thresh

    # Because here one shank is selected per shank, based on visualization:
    # mean: very conservative in cases where some shanks may not have that strong ripple
    # max: works well but may have occasional false positives

    # First, bandpass the signal in the range of interest
    power = np.zeros(signals.shape[1])
    for sig in signals:
        yf = signal_process.filter_sig.bandpass(sig, lf=lf, hf=hf, fs=fs)
        # zsc_chan = smooth(stats.zscore(np.abs(signal_process.hilbertfast(yf))))
        # zscsignal[sig_i] = zsc_chan
        power += np.abs(signal_process.hilbertfast(yf))

    # Second, take the mean and smooth the signal with a sigma wide gaussian kernel
    power = smooth(power / signals.shape[0])

    # Third, exclude any noisy periods due to motion or other artifact
    # ---------setting noisy periods zero --------
    if ignore_times is not None:
        assert ignore_times.ndim == 2, "ignore_times should be 2 dimensional array"
        noisy_frames = np.concatenate(
            [
                (np.arange(start * fs, stop * fs)).astype(int)
                for (start, stop) in ignore_times
            ]
        )

        # edge case: remove any frames that might extend past end of recording
        noisy_frames = noisy_frames[noisy_frames < len(power)]
        power[noisy_frames] = 0

    # Fourth, identify candidate epochs above edge_cutoff threshold
    # ---- thresholding and detection ------
    power = stats.zscore(power)
    # power_thresh = np.where(power >= edge_cutoff, power, 0)
    power_thresh = np.where(power >= edge_cutoff, power, -100)  # NRK bugfix

    # Fifth, refine candidate epochs to periods between lowthresh and highthresh
    peaks, props = sg.find_peaks(
        power_thresh, height=[lowthresh, highthresh], prominence=0
    )
    starts, stops = props["left_bases"], props["right_bases"]
    peaks_power = power_thresh[peaks]

    # ----- merge overlapping epochs ------
    # Last, merge any epochs that overlap into one longer epoch
    n_epochs = len(starts)
    ind_delete = []
    for i in range(n_epochs - 1):
        if starts[i + 1] - stops[i] < 1e-6:
            # stretch the second epoch to cover the range of both epochs
            starts[i + 1] = min(starts[i], starts[i + 1])
            stops[i + 1] = max(stops[i], stops[i + 1])

            peaks_power[i + 1] = max(peaks_power[i], peaks_power[i + 1])
            peaks[i + 1] = [peaks[i], peaks[i + 1]][
                np.argmax([peaks_power[i], peaks_power[i + 1]])
            ]

            ind_delete.append(i)

    epochs_arr = np.vstack((starts, stops, peaks, peaks_power)).T
    starts, stops, peaks, peaks_power = np.delete(epochs_arr, ind_delete, axis=0).T

    epochs_df = pd.DataFrame(
        dict(
            start=starts, stop=stops, peak_time=peaks, peak_power=peaks_power, label=""
        )
    )
    epochs_df[["start", "stop", "peak_time"]] /= fs  # seconds
    epochs = Epoch(epochs=epochs_df)

    # ------duration thresh---------
    epochs = epochs.duration_slice(min_dur=mindur, max_dur=maxdur)
    print(f"{len(epochs)} epochs remaining with durations within ({mindur},{maxdur})")

    epochs.metadata = {
        "params": {
            "lowThres": lowthresh,
            "highThresh": highthresh,
            "freq_band": freq_band,
            "mindur": mindur,
            "maxdur": maxdur
        },
    }
    if not return_power:
        return epochs
    else:
        return epochs, power
    

def _detect_freq_band_epochs(
    signal: Signal,
    filt,
    thresh,
    edge_cutoff,
    sep,
    mindur,
    maxdur,
    sigma_t,
    eff_times=None,
    ignore_times=None,
    return_amp=False,
    verbose=False
):
    """Detects epochs of high power in a given frequency band

    Parameters
    ----------
    signal : Signal
        Input signal object
    filt : tuple
        (low_freq, high_freq) for bandpass filter
    thresh : tuple
        (low_threshold, high_threshold) for detection
    edge_cutoff : float
        Z-score threshold for initial detection
    mindur : float
        Minimum duration of epoch in seconds
    maxdur : float
        Maximum duration of epoch in seconds
    sigma_t : float
        Sigma for Gaussian smoothing in seconds
    eff_times : np.ndarray, optional
        2D array of (start, stop) times to include in seconds (exclusive focus)
    ignore_times : np.ndarray, optional
        2D array of (start, stop) times to ignore in seconds
    return_amp : bool, optional
        If True, return z-scored amplitude envelope along with epochs
    verbose : bool, optional
        Verbosity flag for filtering

    Returns
    -------
    epochs : Epoch
        Detected epochs object
    amp_smoothed_zscored : np.ndarray, optional
        Z-scored amplitude envelope if return_amp=True
    """
    # Work on a copy to avoid modifying original signal
    signal = signal.copy()
    time = signal.time
    fs = signal.sampling_rate
    lf, hf = filt
    dt = 1 / fs
    
    # Define smoothing function
    # Define smoothing function based on sigma_t
    if sigma_t <= 0:
        smooth = lambda x: x
    else:
        smooth = lambda x: gaussian_filter1d(x, sigma=sigma_t / dt, axis=-1)
    lowthresh, highthresh = thresh

    # Step 1: Bandpass filter and extract amplitude envelope
    signal.mne.filter(l_freq=lf, h_freq=hf, verbose=verbose).apply_hilbert(envelope=True)
    signal.update()
    
    # Step 2: Max across channels and smooth
    amp = np.max(signal.traces, axis=0)
    amp_smoothed = smooth(amp)

    # Step 3: Z-score normalization using ALL data (before applying eff_times/ignore_times)
    mean_val = np.mean(amp_smoothed)
    std_val = np.std(amp_smoothed)
    
    if std_val == 0:
        print("Standard deviation is zero, cannot compute z-scores")
        amp_smoothed_zscored = np.zeros_like(amp_smoothed)
    else:
        amp_smoothed_zscored = (amp_smoothed - mean_val) / std_val
    
    # Make a copy for detection that will be masked
    amp_for_detection = amp_smoothed_zscored.copy()

    # Step 4: Apply effective times (keep only specified periods for detection)
    if eff_times is not None:
        assert eff_times.ndim == 2, "eff_times should be 2D array of shape (n_periods, 2)"
        
        # Start with all frames set to NaN (excluded by default)
        mask = np.full(len(amp_for_detection), np.nan)
        
        # Set effective time periods to their actual values
        for start, stop in eff_times:
            # Find closest frame indices using time array
            start_frame = np.argmin(np.abs(time - start))
            stop_frame = np.argmin(np.abs(time - stop))
            mask[start_frame:stop_frame] = amp_for_detection[start_frame:stop_frame]
        
        amp_for_detection = mask

    # Step 5: Exclude noisy periods (overrides eff_times if overlapping)
    if ignore_times is not None:
        assert ignore_times.ndim == 2, "ignore_times should be 2D array of shape (n_periods, 2)"
        
        # Convert time periods to frame indices and set to NaN
        for start, stop in ignore_times:
            # Find closest frame indices using time array
            start_frame = np.argmin(np.abs(time - start))
            stop_frame = np.argmin(np.abs(time - stop))
            amp_for_detection[start_frame:stop_frame] = np.nan

    # Step 6: Check if we have valid data for detection
    valid_mask = ~np.isnan(amp_for_detection)
    if np.sum(valid_mask) == 0:
        print("No valid data remaining after applying eff_times and ignore_times")
        epochs_df = pd.DataFrame(
            columns=['start', 'stop', 'peak_time', 'peak_amp', 'label']
        )
        epochs = Epoch(epochs=epochs_df)
        epochs.metadata = {
            "params": {
                "lowThresh": lowthresh,
                "highThresh": highthresh,
                "freq_band": filt,
                "mindur": mindur,
                "maxdur": maxdur,
            },
        }
        return (epochs, amp_smoothed_zscored) if return_amp else epochs
    
    merged_epochs = detect_epochs_peak(
        signal=amp_for_detection,
        edge_cutoff=edge_cutoff,
        lowthresh=lowthresh,
        highthresh=highthresh,
        prominence=0
    )
    
    if len(merged_epochs) == 0:
        # No epochs detected
        print("No epochs detected with given thresholds")
        epochs_df = pd.DataFrame(
            columns=['start', 'stop', 'peak_time', 'peak_amp', 'label']
        )
        epochs = Epoch(epochs=epochs_df)
        epochs.metadata = {
            "params": {
                "lowThresh": lowthresh,
                "highThresh": highthresh,
                "freq_band": filt,
                "mindur": mindur,
                "maxdur": maxdur,
            },
        }
        return (epochs, amp_smoothed_zscored) if return_amp else epochs
    
    # Unpack merged epochs
    starts, stops, peaks, peaks_amp = merged_epochs.T
    
    # Step 7: Create epochs dataframe with time-based start/stop
    epochs_df = pd.DataFrame({
        'start': time[starts.astype(int)],  # Use time array to get actual timestamps
        'stop': time[stops.astype(int)],    # Use time array to get actual timestamps
        'peak_time': time[peaks.astype(int)],  # Use time array for peak timestamps
        'peak_amp': peaks_amp,
        'label': ""
    })
    
    epochs = Epoch(epochs=epochs_df)
    
    # Step 8: Filter by duration
    if sep != None:
        epochs = epochs.merge_neighbors(max_epoch_sep = sep)
    if mindur != None or maxdur != None:
        epochs = epochs.duration_slice(min_dur=mindur, max_dur=maxdur)
    print(f"{len(epochs)} epochs remaining with durations within ({mindur}, {maxdur})")
    
    # Add metadata
    epochs.metadata = {
        "params": {
            "lowThresh": lowthresh,
            "highThresh": highthresh,
            "freq_band": filt,
            "mindur": mindur,
            "maxdur": maxdur,
        },
    }
    
    return (epochs, amp_smoothed_zscored) if return_amp else epochs

def detect_hpc_delta_wave_epochs(
    signal: Signal,
    freq_band=(0.2, 5),
    min_dur=0.15,
    max_dur=0.5,
    ignore_epochs: Epoch = None,
):
    """Detect delta waves epochs.

    Method
    -------
    Maingret, Nicolas, Gabrielle Girardeau, Ralitsa Todorova, Marie Goutierre, and Michaël Zugaro. “Hippocampo-Cortical Coupling Mediates Memory Consolidation during Sleep.” Nature Neuroscience 19, no. 7 (July 2016): 959–64. https://doi.org/10.1038/nn.4304.

    -> filtered singal in 0.5-4 Hz (Note: Maingret et al. used 0-6 Hz for cortical LFP)
    -> remove noisy epochs if provided
    -> z-scored the filtered signal, D(t)
    -> flip the sign of signal to be consistent with cortical LFP
    -> calculate derivative, D'(t)
    -> extract upward-downward-upward zero-crossings which correspond to start,peak,stop of delta waves, t_start, t_peak, t_stop
    -> discard sequences below 150ms and above 500ms
    -> Delta waves corresponded to epochs where D(t_peak) > 2, or D(t_peak) > 1 and D(t_stop) < -1.5.

    Parameters
    ----------
    signal : Signal object
        signal trace to be used for detection
    freq_band : tuple, optional
        frequency band in Hz, by default (0.5, 4)
    min_dur: float, optional
        minimum duration for delta waves, by default 0.15 seconds
    max_dur: float, optional
        maximum duration for delta waves, by default 0.5 seconds
    ignore_epochs: core.Epoch, optional
        ignore timepoints within these epochs, primarily used for noisy time periods if known already, by default None

    Returns
    -------
    Epoch
        delta wave epochs. In addition, peak_time, peak_amp_zsc, stop_amp_zsc are also returned as columns
    """

    assert freq_band[1] <= 6, "Upper limit of freq_band can not be above 6 Hz"
    assert signal.n_channels == 1, "Signal should have only 1 channel"

    delta_signal = signal_process.filter_sig.bandpass(
        signal, lf=freq_band[0], hf=freq_band[1]
    ).traces[0]
    time = signal.time

    # ----- remove timepoints provided in ignore_epochs ------
    if ignore_epochs is not None:
        noisy_bool = ignore_epochs.get_indices_for_time(time)
        time = time[~noisy_bool]
        delta_signal = delta_signal[~noisy_bool]

    # ---- normalize and flip the sign to be consistent with cortical lfp ----
    delta_zsc = -1 * stats.zscore(delta_signal)

    # ---- finding peaks and trough for delta oscillations
    delta_zsc_diff = np.diff(delta_zsc).squeeze()
    zero_crossings = np.diff(np.sign(delta_zsc_diff))
    troughs_indx = np.where(zero_crossings > 0)[0]
    peaks_indx = np.where(zero_crossings < 0)[0]

    if peaks_indx[0] < troughs_indx[0]:
        peaks_indx = peaks_indx[1:]

    if peaks_indx[-1] > troughs_indx[-1]:
        peaks_indx = peaks_indx[:-1]

    n_peaks_in_troughs = np.histogram(peaks_indx, troughs_indx)[0]
    assert n_peaks_in_troughs.max() == 1, "Found multiple peaks within troughs"

    troughs_time, peaks_time = time[troughs_indx], time[peaks_indx]
    trough_pairs = np.vstack((troughs_time[:-1], troughs_time[1:])).T
    trough_peak_trough = np.insert(trough_pairs, 1, peaks_time, axis=1)
    duration = np.diff(trough_pairs, axis=1).squeeze()
    peak_amp = delta_zsc[peaks_indx]
    stop_amp = delta_zsc[troughs_indx[1:]]

    # ---- filtering based on duration and z-scored amplitude -------
    good_duration_bool = (duration >= min_dur) & (duration <= max_dur)
    good_amp_bool = (peak_amp > 2) | ((peak_amp > 1.5) & (stop_amp < -1.5))
    good_bool = good_amp_bool & good_duration_bool

    delta_waves_time = trough_peak_trough[good_bool]

    print(f"{delta_waves_time.shape[0]} delta waves detected")

    epochs = pd.DataFrame(
        {
            "start": delta_waves_time[:, 0],
            "stop": delta_waves_time[:, 2],
            "peak_time": delta_waves_time[:, 1],
            "peak_amp_zsc": peak_amp[good_bool],
            "stop_amp_zsc": stop_amp[good_bool],
            "label": "delta_wave",
        }
    )
    params = {"freq_band": freq_band, "channel": signal.channel_id}

    return Epoch(epochs=epochs, metadata=params)


def detect_beta_epochs(
    signal: Signal,
    probegroup: ProbeGroup = None,
    freq_band=(15, 40),
    thresh=(0, 0.5),
    mindur=0.25,
    maxdur=5,
    mergedist=0.5,
    sigma=0.125,
    edge_cutoff=-0.25,
    ignore_epochs: Epoch = None,
    return_power=False,
):
    if probegroup is None:
        selected_chan = signal.channel_id
        traces = signal.traces
    else:
        if isinstance(probegroup, np.ndarray):
            changrps = np.array(probegroup, dtype="object")
        if isinstance(probegroup, ProbeGroup):
            changrps = probegroup.get_connected_channels(groupby="shank")
        channel_ids = np.concatenate(changrps).astype("int")

        duration = signal.duration
        t1, t2 = signal.t_start, signal.t_start + np.min([duration, 3600])
        signal_slice = signal.time_slice(channel_id=channel_ids, t_start=t1, t_stop=t2)
        hil_stat = signal_process.hilbert_amplitude_stat(
            signal_slice.traces,
            freq_band=freq_band,
            fs=signal.sampling_rate,
            statistic="mean",
        )
        selected_chan = channel_ids[np.argmax(hil_stat)].reshape(-1)
        traces = signal.time_slice(channel_id=selected_chan).traces.reshape(1, -1)

    print(f"Best channel for beta: {selected_chan}")
    if ignore_epochs is not None:
        ignore_times = ignore_epochs.as_array()
    else:
        ignore_times = None

    epochs = _detect_freq_band_epochs(
        signals=traces,
        freq_band=freq_band,
        thresh=thresh,
        mindur=mindur,
        maxdur=maxdur,
        mergedist=mergedist,
        fs=signal.sampling_rate,
        ignore_times=ignore_times,
        sigma=sigma,
        edge_cutoff=edge_cutoff,
        return_power=return_power,
    )

    if not return_power:
        epochs = epochs.shift(dt=signal.t_start)
        epochs.metadata = dict(channels=selected_chan)
        return epochs
    else:
        beta_power = epochs[1]
        epochs = epochs[0].shift(dt=signal.t_start)
        epochs.metadata = dict(channels=selected_chan)
        return epochs, beta_power


   
def detect_ripple_epochs(
    signal: Signal,
    probegroup: ProbeGroup = None,
    freq_band=(150, 250),
    thresh=(2.5, None),
    edge_cutoff=0.5,
    mindur=None,
    maxdur=None,
    sep=None,
    sigma=0.0125,
    eff_epochs: Epoch = None,
    ignore_epochs: Epoch = None,
    ripple_channel: int or list = None,
    return_amp: bool = False,
    verbose=False,
):
    """Detect sharp-wave ripple epochs in neural signal data.
    
    Parameters
    ----------
    signal : Signal
        Input signal object
    probegroup : ProbeGroup or np.ndarray, optional
        Probe configuration for channel selection. If None, uses all channels.
    freq_band : tuple, optional
        (low_freq, high_freq) for ripple band, default (150, 250) Hz
    thresh : tuple, optional
        (low_threshold, high_threshold) z-score thresholds for detection
    edge_cutoff : float, optional
        Z-score cutoff for edge detection, default 0.5
    mindur : float, optional
        Minimum epoch duration in seconds, default 0.05
    maxdur : float, optional
        Maximum epoch duration in seconds, default 0.450
    sep : float, optional
        Maximum separation for merging nearby epochs in seconds, default 0.05
    sigma : float, optional
        Gaussian smoothing sigma in seconds, default 0.0125
    eff_epochs : Epoch, optional
        Epochs to restrict detection to (exclusive focus)
    ignore_epochs : Epoch, optional
        Epochs to exclude from detection
    ripple_channel : int or list, optional
        Specific channel(s) to use for detection. If None, auto-detects best channel.
    return_amp : bool, optional
        If True, also return ripple power envelope
    verbose : bool, optional_detect_freq_band_epochs
        Verbosity flag for filtering
        
    Returns
    -------
    epochs : Epoch
        Detected ripple epochs
    ripple_power : np.ndarray, optional
        Ripple power envelope if return_amp=True
        
    Notes
    -----
    TODO: Add chewing artifact rejection (>300 Hz) or EMG-based filtering
    """
    signal = signal.copy()

    # Step 1: Channel selection
    if ripple_channel is not None:
        # Use user-specified channel(s)
        selected_chans = (
            [ripple_channel] if isinstance(ripple_channel, int) else ripple_channel
        )
        signal_ch = signal.time_slice(channel_id=selected_chans)
    else:
        # Auto-detect best ripple channel(s)
        if probegroup is None:
            # Use all available channels
            selected_chans = signal.channel_id
            signal_ch = signal
        else:
            # Parse probe group configuration
            if isinstance(probegroup, np.ndarray):
                changrps = np.array(probegroup, dtype="object")
            elif isinstance(probegroup, ProbeGroup):
                changrps = probegroup.get_connected_channels(groupby="shank")
            else:
                raise TypeError(f"probegroup must be ProbeGroup, np.ndarray, or None, got {type(probegroup)}")
            
            # Select best channel per shank based on ripple power
            selected_chans = []
            for changrp in changrps:
                # Use first hour (or full duration if shorter) for channel selection
                duration = signal.duration
                t_stop = min(duration, 3600)
                
                # Get signal slice for this channel group
                signal_slice = signal.time_slice(
                    channel_id=changrp.astype("int"),
                    t_start=signal.t_start,
                    t_stop=signal.t_start + t_stop,
                )
                
                # Calculate ripple power across channels
                hil_stat = signal_process.hilbert_amplitude_stat(
                    signal_slice.traces,
                    freq_band=freq_band,
                    fs=signal.sampling_rate,
                    statistic="mean",
                )
                
                # Select channel with highest mean ripple power
                best_chan = changrp[np.argmax(hil_stat)]
                selected_chans.append(best_chan)
            
            signal_ch = signal.time_slice(channel_id=selected_chans)
    
    print(f"Selected channels for ripples: {selected_chans}")
    
    # Step 2: Prepare time constraints
    ignore_times = ignore_epochs.as_array() if ignore_epochs is not None else None
    eff_times = eff_epochs.as_array() if eff_epochs is not None else None
    
    # Step 3: Detect ripple epochs
    result = _detect_freq_band_epochs(
        signal=signal_ch,
        filt=freq_band,
        thresh=thresh,
        sep=sep,
        mindur=mindur,
        maxdur=maxdur,
        eff_times=eff_times,
        ignore_times=ignore_times,
        sigma_t=sigma,
        edge_cutoff=edge_cutoff,
        return_amp=return_amp,
        verbose=verbose
    )
    
    # Step 4: Package results with metadata
    if return_amp:
        epochs, ripple_power = result
        epochs.metadata = dict(channels=selected_chans)
        return epochs, ripple_power
    else:
        epochs = result
        epochs.metadata = dict(channels=selected_chans)
        return epochs



def detect_sharpwave_epochs(
    signal: Signal,
    probegroup: ProbeGroup = None,
    freq_band=(2, 50),
    thresh=(2.5, None),
    edge_cutoff=0.5,
    mindur=0.05,
    maxdur=0.450,
    mergedist=0.05,
    sigma=0.0125,
    ignore_epochs: Epoch = None,
    sharpwave_channel: int or list = None,
):
    if sharpwave_channel is None:
        if probegroup is None:  # auto-detect sharpwave channel
            selected_chans = signal.channel_id
            traces = signal.traces

        else:
            if isinstance(probegroup, np.ndarray):
                changrps = np.array(probegroup, dtype="object")
            if isinstance(probegroup, ProbeGroup):
                changrps = probegroup.get_connected_channels(groupby="shank")
                # if changrp:
            selected_chans = []
            for changrp in changrps:
                signal_slice = signal.time_slice(
                    channel_id=changrp.astype("int"),
                    t_start=0,
                    t_stop=np.min((3600, signal.duration)),
                )
                hil_stat = signal_process.hilbert_amplitude_stat(
                    signal_slice.traces,
                    freq_band=freq_band,
                    fs=signal.sampling_rate,
                    statistic="mean",
                )
                selected_chans.append(changrp[np.argmax(hil_stat)])

            traces = signal.time_slice(channel_id=selected_chans).traces
    else:
        assert isinstance(sharpwave_channel, (list, int))
        selected_chans = (
            [sharpwave_channel]
            if isinstance(sharpwave_channel, int)
            else sharpwave_channel
        )
        traces = signal.time_slice(channel_id=selected_chans).traces

    print(f"Selected channels for sharp-waves: {selected_chans}")
    if ignore_epochs is not None:
        ignore_times = ignore_epochs.as_array()
    else:
        ignore_times = None

    epochs = _detect_freq_band_epochs(
        signals=traces,
        freq_band=freq_band,
        thresh=thresh,
        edge_cutoff=edge_cutoff,
        mindur=mindur,
        maxdur=maxdur,
        mergedist=mergedist,
        fs=signal.sampling_rate,
        sigma=sigma,
        ignore_times=ignore_times,
    )
    epochs = epochs.shift(dt=signal.t_start)
    epochs.metadata = dict(channels=selected_chans)
    return epochs

def detect_theta_epochs(
    signal: Signal,
    probegroup: ProbeGroup = None,
    freq_band=(5, 12),
    thresh=(0, 0.5),
    sep = 0.2,
    mindur=0.25,
    maxdur=5,
    sigma=0.125,
    edge_cutoff=-0.25,
    eff_epochs: Epoch = None,
    ignore_epochs: Epoch = None,
    return_amp=False,
    theta_channel: int or list = None,
    verbose=False
):
    """Detect theta oscillation epochs in neural signal data.
    
    Parameters
    ----------
    signal : Signal
        Input signal object
    probegroup : ProbeGroup or np.ndarray, optional
        Probe configuration for channel selection. If None, uses all channels.
    freq_band : tuple, optional
        (low_freq, high_freq) for theta band, default (5, 12) Hz
    thresh : tuple, optional
        (low_threshold, high_threshold) z-score thresholds for detection
    mindur : float, optional
        Minimum epoch duration in seconds, default 0.25
    maxdur : float, optional
        Maximum epoch duration in seconds, default 5
    mergedist : float, optional
        Distance for merging nearby epochs (currently unused)
    sigma : float, optional
        Gaussian smoothing sigma in seconds, default 0.125
    edge_cutoff : float, optional
        Z-score cutoff for edge detection, default -0.25
    eff_epochs : Epoch, optional
        Epochs to restrict detection to (exclusive focus)
    ignore_epochs : Epoch, optional
        Epochs to exclude from detection
    return_power : bool, optional
        If True, also return theta power envelope
    verbose : bool, optional
        Verbosity flag for filtering
        
    Returns
    -------
    epochs : Epoch
        Detected theta epochs
    theta_power : np.ndarray, optional
        Theta power envelope if return_power=True
    """
    # Work on a copy to avoid modifying original
    signal = signal.copy()
    
    # Step 1: Channel selection
    if theta_channel is not None:
        # Use specified theta channel(s)
        selected_chan = (
            [theta_channel] if isinstance(theta_channel, int) else theta_channel
        )
        signal_ch = signal.time_slice(channel_id=selected_chan)
        print(f"Best channel for theta: {selected_chan}")
    else:
        if probegroup is None:
            # Use all available channels
            selected_chan = signal.channel_id
            signal_ch = signal
        else:
            # Parse probe group configuration
            if isinstance(probegroup, np.ndarray):
                changrps = np.array(probegroup, dtype="object")
            elif isinstance(probegroup, ProbeGroup):
                changrps = probegroup.get_connected_channels(groupby="shank")
            else:
                raise TypeError(f"probegroup must be ProbeGroup, np.ndarray, or None, got {type(probegroup)}")
            
            # Get all channel IDs from probe groups
            channel_ids = np.concatenate(changrps).astype("int")
            
            # Select best channel based on theta power in first hour (or full duration if shorter)
            duration = signal.duration
            t1 = signal.t_start
            t2 = signal.t_start + min(duration, 3600)  # Use min() instead of np.min()
            
            # Get signal slice for channel selection
            signal_slice = signal.time_slice(channel_id=channel_ids, t_start=t1, t_stop=t2)
            
            # Calculate theta power across channels
            hil_stat = signal_process.hilbert_amplitude_stat(
                signal_slice,
                freq_band=freq_band,
                statistic="mean",
            )
            
            # Select channel with highest mean theta power
            selected_chan = channel_ids[np.argmax(hil_stat)].reshape(-1)
            signal_ch = signal.time_slice(channel_id=selected_chan)
            
            print(f"Best channel for theta: {selected_chan}")
    
    # Step 2: Prepare time constraints
    ignore_times = ignore_epochs.as_array() if ignore_epochs is not None else None
    eff_times = eff_epochs.as_array() if eff_epochs is not None else None
    
    # Step 3: Detect theta epochs
    result = _detect_freq_band_epochs(
        signal=signal_ch,
        filt=freq_band,
        thresh=thresh,
        sep=sep,
        mindur=mindur,
        maxdur=maxdur,
        eff_times=eff_times,
        ignore_times=ignore_times,
        sigma_t=sigma,
        edge_cutoff=edge_cutoff,
        return_amp=return_amp,
        verbose=verbose
    )
    
    # Step 4: Process results and adjust timing to original signal reference
    if return_amp:
        epochs, theta_power = result
        epochs.metadata = dict(channels=selected_chan)
        return epochs, theta_power
    else:
        epochs = result
        epochs.metadata = dict(channels=selected_chan)
        return epochs

def detect_spindle_epochs(
    signal: Signal,
    probegroup: ProbeGroup = None,
    freq_band=(8, 16),
    thresh=(1, 5),
    mindur=0.35,
    maxdur=4,
    mergedist=0.05,
    ignore_epochs: Epoch = None,
    method="hilbert",
):
    if probegroup is None:
        selected_chans = signal.channel_id
        traces = signal.traces

    else:
        if isinstance(probegroup, np.ndarray):
            changrps = np.array(probegroup, dtype="object")
        if isinstance(probegroup, ProbeGroup):
            changrps = probegroup.get_connected_channels(groupby="shank")
            # if changrp:
        selected_chans = []
        for changrp in changrps:
            signal_slice = signal.time_slice(
                channel_id=changrp.astype("int"), t_start=0, t_stop=3600
            )
            hil_stat = signal_process.hilbert_amplitude_stat(
                signal_slice.traces,
                freq_band=freq_band,
                fs=signal.sampling_rate,
                statistic="mean",
            )
            selected_chans.append(changrp[np.argmax(hil_stat)])

        traces = signal.time_slice(channel_id=selected_chans).traces

    print(f"Selected channels for spindles: {selected_chans}")

    if ignore_epochs is not None:
        ignore_times = ignore_epochs.as_array()
    else:
        ignore_times = None

    epochs, metadata = _detect_freq_band_epochs(
        signals=traces,
        freq_band=freq_band,
        thresh=thresh,
        mindur=mindur,
        maxdur=maxdur,
        mergedist=mergedist,
        fs=signal.sampling_rate,
        ignore_times=ignore_times,
    )
    epochs["start"] = epochs["start"] + signal.t_start
    epochs["stop"] = epochs["stop"] + signal.t_start

    metadata["channels"] = selected_chans
    return Epoch(epochs=epochs, metadata=metadata)


def detect_gamma_epochs():
    pass


class Ripple:
    """Events and analysis related to sharp-wave ripple oscillations"""

    @staticmethod
    def detect_ripple_epochs(**kwargs):
        return detect_ripple_epochs(**kwargs)

    @staticmethod
    def detect_sharpwave_epochs(**kwargs):
        return detect_sharpwave_epochs(**kwargs)

    @staticmethod
    def get_mean_wavelet(eegfile, rpl_channel, rpl_epochs, lf=100, hf=300, buffer_sec=0.1):

        # Load in relevant metadata
        sampling_rate = eegfile.sampling_rate

        # Specify frequencies and get signal
        freqs = np.linspace(100, 250, 100)
        signal = eegfile.get_signal(channel_indx=rpl_channel)

        # Bandpass signal in ripple range
        lfp = filter_sig.bandpass(signal, lf=lf, hf=hf).traces.mean(axis=0)

        # Build up arrays to grab every 1000 ripples
        n_rpls = len(rpl_epochs)
        rpls_window = np.arange(0, n_rpls, np.min([1000, n_rpls - 1]))
        rpls_window[-1] = n_rpls
        peak_freqs, peak_power = [], []

        # Loop through each set of 1000 ripples, concatenate signal for each together, run Wavelet and get peak frequency
        # at time of peak power
        buffer_frames = int(buffer_sec * sampling_rate)  # grab 100ms either side of peak power
        sxx_mean, wvlt_n = [], []
        for i in range(len(rpls_window) - 1):
            # Get blocks of ripples and their peak times
            rpl_df = rpl_epochs[rpls_window[i] : rpls_window[i + 1]].to_dataframe()
            peakframe = (rpl_df["peak_time"].values * sampling_rate).astype("int")

            rpl_frames = [np.arange(p - buffer_frames, p + buffer_frames) for p in peakframe]  # Grab 100ms either side of peak frame
            rpl_frames = np.concatenate(rpl_frames)

            # Grab signal for ripples only
            new_sig = Signal(lfp[rpl_frames].reshape(1, -1), sampling_rate=sampling_rate)

            # Run Wavelet and get peak frequency for each ripple
            wvlt = WaveletSg(signal=new_sig, freqs=freqs, ncycles=10)
            sxx_mean.append(np.reshape(wvlt.traces, (len(freqs), len(peakframe), -1)).mean(axis=1)[:, :, None])
            wvlt_n.append(rpl_df.shape[0])

        # Get mean wavelet
        sxx_mean = np.concatenate(sxx_mean, axis=2)
        sxx_mean = np.average(sxx_mean, axis=2, weights=wvlt_n)

        # Make into Wavelet class
        wvlt_mean = deepcopy(wvlt)
        wvlt_mean.traces = sxx_mean
        wvlt_mean.t_start = -buffer_sec

        return wvlt_mean


    @staticmethod
    def get_peak_ripple_freq(eegfile: BinarysignalIO, rpl_channel, rpl_epochs: Epoch, lf=100, hf=300):
        """Detect peak ripple frequency"""

        # Load in relevant metadata
        sampling_rate = eegfile.sampling_rate

        # Specify frequencies and get signal
        freqs = np.linspace(100, 250, 100)
        signal = eegfile.get_signal(channel_indx=rpl_channel)

        # Bandpass signal in ripple range
        lfp = filter_sig.bandpass(signal, lf=lf, hf=hf).traces.mean(axis=0)

        # Build up arrays to grab every 1000 ripples
        n_rpls = len(rpl_epochs)
        rpls_window = np.arange(0, n_rpls, np.min([1000, n_rpls - 1]))
        rpls_window[-1] = n_rpls
        peak_freqs, peak_power = [], []

        # Loop through each set of 1000 ripples, concatenate signal for each together, run Wavelet and get peak frequency
        # at time of peak power
        buffer_frames = int(.1 * sampling_rate)  # grab 100ms either side of peak power
        for i in range(len(rpls_window) - 1):
            # Get blocks of ripples and their peak times
            rpl_df = rpl_epochs[rpls_window[i] : rpls_window[i + 1]].to_dataframe()
            peakframe = (rpl_df["peak_time"].values * sampling_rate).astype("int")

            rpl_frames = [np.arange(p - buffer_frames, p + buffer_frames) for p in peakframe]  # Grab 100ms either side of peak frame
            rpl_frames = np.concatenate(rpl_frames)

            # Grab signal for ripples only
            new_sig = Signal(lfp[rpl_frames].reshape(1, -1), sampling_rate=sampling_rate)

            # Run Wavelet and get peak frequency for each ripple
            wvlt = WaveletSg(signal=new_sig, freqs=freqs, ncycles=10).traces
            peak_freqs.append(
                freqs[
                    np.reshape(wvlt, (len(freqs), len(peakframe), -1))
                    .max(axis=2)
                    .argmax(axis=0)
                ]
            )
            peak_power.append(np.reshape(wvlt, (len(freqs), len(peakframe), -1))
                              .max(axis=2)
                              .max(axis=0))

        # Concatenate all peak frequencies found
        peak_freqs = np.concatenate(peak_freqs)
        peak_power = np.concatenate(peak_power)
        assert len(peak_freqs) == len(rpl_epochs), "# peak frequencies found does not match size of input 'rpl_epochs', check code"
        new_epochs = rpl_epochs.add_column("peak_frequency_bp", peak_freqs)
        new_epochs = new_epochs.add_column("peak_power", peak_power)

        return new_epochs


class Gamma:
    """Events and analysis related to gamma oscillations"""

    def get_peak_intervals(
        self,
        lfp,
        band=(40, 80),
        lowthresh=0,
        highthresh=1,
        minDistance=300,
        minDuration=125,
        return_amplitude=False,

    ):
        """Returns strong theta lfp. If it has multiple channels, then strong theta periods are calculated from that
        channel which has highest area under the curve in the theta frequency band. Parameters are applied on z-scored lfp.

        Parameters
        ----------
        lfp : array like, channels x time
            from which strong periods are concatenated and returned
        lowthresh : float, optional
            threshold above which it is considered strong, by default 0 which is mean of the selected channel
        highthresh : float, optional
            [description], by default 0.5
        minDistance : int, optional
            minimum gap between periods before they are merged, by default 300 samples
        minDuration : int, optional
            [description], by default 1250, which means theta period should atleast last for 1 second

        Returns
        -------
        2D array
            start and end frames where events exceeded the set thresholds
        """

        # ---- filtering --> zscore --> threshold --> strong gamma periods ----
        gammalfp = signal_process.filter_sig.bandpass(lfp, lf=band[0], hf=band[1])
        hil_gamma = signal_process.hilbertfast(gammalfp)
        gamma_amp = np.abs(hil_gamma)

        zsc_gamma = stats.zscore(gamma_amp)
        peakevents = mathutil.threshPeriods(
            zsc_gamma,
            lowthresh=lowthresh,
            highthresh=highthresh,
            minDistance=minDistance,
            minDuration=minDuration,
        )
        if not return_amplitude:
            return peakevents
        else:
            return peakevents, gamma_amp

    def csd(self, period, refchan, chans, band=(40, 80), window=1250):
        """Calculating current source density using laplacian method

        Parameters
        ----------
        period : array
            period over which theta cycles are averaged
        refchan : int or array
            channel whose theta peak will be considered. If array then median of lfp across all channels will be chosen for peak detection
        chans : array
            channels for lfp data
        window : int, optional
            time window around theta peak in number of samples, by default 1250

        Returns:
        ----------
        csd : dataclass,
            a dataclass return from signal_process module
        """
        lfp_period = self._obj.geteeg(chans=chans, timeRange=period)
        lfp_period = signal_process.filter_sig.bandpass(
            lfp_period, lf=band[0], hf=band[1]
        )

        gamma_lfp = self._obj.geteeg(chans=refchan, timeRange=period)
        nChans = lfp_period.shape[0]
        # lfp_period, _, _ = self.getstrongTheta(lfp_period)

        # --- Selecting channel with strongest theta for calculating theta peak-----
        # chan_order = self._getAUC(lfp_period)
        # gamma_lfp = signal_process.filter_sig.bandpass(
        #     lfp_period[chan_order[0], :], lf=5, hf=12, ax=-1)
        gamma_lfp = signal_process.filter_sig.bandpass(
            gamma_lfp, lf=band[0], hf=band[1]
        )
        peak = sg.find_peaks(gamma_lfp)[0]
        # Ignoring first and last second of data
        peak = peak[np.where((peak > 1250) & (peak < len(gamma_lfp) - 1250))[0]]

        # ---- averaging around theta cycle ---------------
        avg_theta = np.zeros((nChans, window))
        for ind in peak:
            avg_theta = avg_theta + lfp_period[:, ind - window // 2 : ind + window // 2]
        avg_theta = avg_theta / len(peak)

        _, ycoord = self._obj.probemap.get(chans=chans)

        csd = signal_process.Csd(lfp=avg_theta, coords=ycoord, chan_label=chans)
        csd.classic()

        return csd


if __name__ == "__main__":
    from neuropy.io import BinarysignalIO
    from neuropy.core import ProbeGroup, Epoch

    eegfile = BinarysignalIO("/data2/Opto/Jackie671/Jackie_propofol_2020-09-30/Jackie_propofol.eeg",
                             n_channels=35, sampling_rate=1250)
    signal = eegfile.get_signal()
    ripple_epochs = Epoch(epochs=None, file="/data2/Opto/Jackie671/Jackie_propofol_2020-09-30/Jackie_propofol.ripple.npy")
    Ripple.get_mean_wavelet(eegfile, rpl_channel=30, rpl_epochs=ripple_epochs)
