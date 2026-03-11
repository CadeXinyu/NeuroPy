import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.signal import find_peaks
from ..utils import mathutil
from .. import core
from neuropy.core.epoch import Epoch
from scipy.ndimage import gaussian_filter1d
from ..utils.detect import detect_epochs_peak


def detect_off_epochs(mua: core.Mua, ignore_epochs: core.Epoch = None):
    """Detects OFF periods using multiunit activity. During these epochs neurons stop almost stop firing.
    These off periods were reported by Vyazovskiy et al. 2011 in cortex for sleep deprived animals.

    Parameters
    ----------
    mua : core.Mua object
        mua object holds total number of spikes in each bin
    ignore_epochs: core.Epoch
        ignore these epochs from getting detected

    References
    ----------
    1) Vyazovskiy, V. V., Olcese, U., Hanlon, E. C., Nir, Y., Cirelli, C., & Tononi, G. (2011). Local sleep in awake rats. Nature, 472(7344), 443–447. https://doi.org/10.1038/nature10009


    """

    time = mua.time
    frate = mua.firing_rate

    # off periods
    off = np.diff(np.where(frate < np.median(frate), 1, 0))
    start_off = np.where(off == 1)[0]
    end_off = np.where(off == -1)[0]

    if start_off[0] > end_off[0]:
        end_off = end_off[1:]
    if start_off[-1] > end_off[-1]:
        start_off = start_off[:-1]

    offperiods = np.vstack((start_off, end_off)).T
    duration = np.diff(offperiods, axis=1).squeeze()

    # ---- calculate minimum instantenous frate within intervals ------
    minValue = np.zeros(len(offperiods))
    for i in range(0, len(offperiods)):
        minValue[i] = min(frate[offperiods[i, 0] : offperiods[i, 1]])

    # --- selecting only top 10 percent of lowest peak instfiring -----
    quantiles = pd.qcut(minValue, 10, labels=False)
    top10percent = np.where(quantiles == 0)[0]
    offperiods = offperiods[top10percent, :]
    duration = duration[top10percent]

    events = pd.DataFrame(
        {
            "start": time[offperiods[:, 0]],
            "stop": time[offperiods[:, 1]],
            "duration": duration * mua.bin_size,
            "label": "",
        }
    )

    return core.Epoch(events)


# def detect_pbe_epochs(
#     mua: core.Mua,
#     thresh=(3, None),
#     edge_cutoff=0.5,
#     duration=(0.1, None),
#     distance=None,
# ):
#     """Detects putative population burst events

#     Parameters
#     ----------
#     thresh : tuple, optional
#         values based on zscore i.e, events with firing rate above thresh[0] and peak exceeding thresh[1], by default (0, 3) --> above mean and greater than 3 SD
#     duration : float, optional
#         minimum and maximum duration of pbe, in seconds, default = (0.1,None) seconds
#     distance : float, optioal
#         if two events are less than this time apart, they are merged, in seconds
#     """

#     assert len(thresh) == 2, "thresh can only have two elements"
#     if distance is None:
#         distance = 1e-6
#     else:
#         distance = distance / mua.bin_size

#     min_dur, max_dur = duration
#     params = {
#         "thresh": thresh,
#         "duration": duration,
#         "distance": distance,
#     }

#     lowthresh, highthresh = thresh
#     n_spikes = stats.zscore(mua.spike_counts)
#     n_spikes_thresh = np.where(n_spikes >= edge_cutoff, n_spikes, 0)
#     peaks, props = find_peaks(
#         n_spikes_thresh, height=[lowthresh, highthresh], prominence=0
#     )
#     starts, stops = props["left_bases"], props["right_bases"]
#     peaks_n_spikes = n_spikes_thresh[peaks]

#     # ----- merge overlapping epochs ------
#     n_epochs = len(starts)
#     ind_delete = []
#     for i in range(n_epochs - 1):
#         if starts[i + 1] - stops[i] < distance:

#             # stretch the second epoch to cover the range of both epochs
#             starts[i + 1] = min(starts[i], starts[i + 1])
#             stops[i + 1] = max(stops[i], stops[i + 1])

#             peaks_n_spikes[i + 1] = max(peaks_n_spikes[i], peaks_n_spikes[i + 1])
#             peaks[i + 1] = [peaks[i], peaks[i + 1]][
#                 np.argmax([peaks_n_spikes[i], peaks_n_spikes[i + 1]])
#             ]

#             ind_delete.append(i)

#     epochs_arr = np.vstack((starts, stops, peaks, peaks_n_spikes)).T
#     starts, stops, peaks, peaks_n_spikes = np.delete(epochs_arr, ind_delete, axis=0).T

#     time = np.asarray(mua.time)
#     epochs_df = pd.DataFrame(
#         {
#             "start": time[starts.astype("int")],
#             "stop": time[stops.astype("int")],
#             "peak_time": time[peaks.astype("int")],
#             "peak_counts": peaks_n_spikes,
#             "label": "pbe",
#         }
#     )
#     epochs = core.Epoch(epochs=epochs_df)
#     # ------duration thresh---------
#     epochs = epochs.duration_slice(min_dur=min_dur, max_dur=max_dur)
#     print(f"{len(epochs)} epochs reamining with durations within ({min_dur},{max_dur})")

#     epochs.metadata = params
#     return epochs


def detect_pbe_epochs(
    mua: core.Mua,
    thresh=(3, None),
    edge_cutoff=0.5,
    duration=(0.1, None),
    sep=0.0,
    sigma_t=0.0,
    eff_epochs: Epoch = None,
    ignore_epochs: Epoch = None,
):
    """Detects putative population burst events (PBEs) from multi-unit activity.
    
    Parameters
    ----------
    mua : core.Mua
        Multi-unit activity object containing spike counts
    thresh : tuple, optional
        (low_threshold, high_threshold) z-score thresholds for detection.
        Events must have firing rate above thresh[0] and peak exceeding thresh[1].
        Default (3, None) means events above 3 SD
    edge_cutoff : float, optional
        Z-score cutoff for edge detection, default 0.5
    duration : tuple, optional
        (min, max) duration of PBE in seconds, default (0.1, None)
    sep : float, optional
        Maximum separation for merging nearby epochs in seconds, default 0.0
    sigma_t : float, optional
        Gaussian smoothing sigma in seconds, default 0.0 (no smoothing)
    eff_epochs : Epoch, optional
        Epochs to restrict detection to (exclusive focus)
    ignore_epochs : Epoch, optional
        Epochs to exclude from detection
        
    Returns
    -------
    Epoch
        Detected PBE epochs with metadata
    """
    # Store parameters as metadata
    metadata = {
        'thresh': thresh,
        'edge_cutoff': edge_cutoff,
        'duration': duration,
        'sep': sep,
        'sigma_t': sigma_t,
    }
    
    lowthresh, highthresh = thresh
    min_dur, max_dur = duration
    dt = mua.bin_size
    time = np.asarray(mua.time)
    
    # Step 1: Apply smoothing if requested
    if sigma_t > 0:
        spike_counts_smoothed = gaussian_filter1d(
            mua.firing_rate, 
            sigma=sigma_t / dt
        )
    else:
        spike_counts_smoothed = mua.firing_rate

    # # Step 2: Z-score normalization of (possibly smoothed) spike counts
    # spike_counts_zscored = stats.zscore(spike_counts_smoothed)
    
    # # Make a copy for detection that will be masked
    signal_for_detection = spike_counts_smoothed.copy()
    
    # Step 3: Apply effective times (keep only specified periods for detection)
    if eff_epochs is not None:
        eff_times = eff_epochs.as_array()
        
        # Start with all frames set to NaN (excluded by default)
        mask = np.full(len(signal_for_detection), np.nan)
        
        # Set effective time periods to their actual values
        for start, stop in eff_times:
            # Find closest frame indices using time array
            start_frame = np.argmin(np.abs(time - start))
            stop_frame = np.argmin(np.abs(time - stop))
            mask[start_frame:stop_frame] = signal_for_detection[start_frame:stop_frame]
        
        signal_for_detection = mask
    
    # Step 4: Exclude noisy periods (overrides eff_epochs if overlapping)
    if ignore_epochs is not None:
        ignore_times = ignore_epochs.as_array()
        
        # Convert time periods to frame indices and set to NaN
        for start, stop in ignore_times:
            # Find closest frame indices using time array
            start_frame = np.argmin(np.abs(time - start))
            stop_frame = np.argmin(np.abs(time - stop))
            signal_for_detection[start_frame:stop_frame] = np.nan
    
    # Step 5: Check if we have valid data for detection
    valid_mask = ~np.isnan(signal_for_detection)
    if np.sum(valid_mask) == 0:
        print("No valid data remaining after applying eff_epochs and ignore_epochs")
        epochs_df = pd.DataFrame(
            columns=['start', 'stop', 'peak_time', 'peak_counts', 'label']
        )
        epochs = Epoch(epochs=epochs_df, metadata=metadata)
        return epochs
    
    # Step 6: Detect PBE epochs using detect_epochs_peak
    merged_epochs = detect_epochs_peak(
        signal=signal_for_detection,
        edge_cutoff=edge_cutoff,
        lowthresh=lowthresh,
        highthresh=highthresh,
        prominence=0
    )
    
    # Check if any epochs were detected
    if len(merged_epochs) == 0:
        print("No PBE epochs detected with given thresholds")
        epochs_df = pd.DataFrame(
            columns=['start', 'stop', 'peak_time', 'peak_counts', 'label']
        )
        epochs = Epoch(epochs=epochs_df, metadata=metadata)
        return epochs
    
    # Step 7: Unpack epochs array
    start_inds, stop_inds, peak_inds, peak_counts = merged_epochs.T
    
    # Step 8: Convert indices to time
    starts_time = time[start_inds.astype('int')]
    stops_time = time[stop_inds.astype('int')]
    peaks_time = time[peak_inds.astype('int')]
    
    # Step 9: Create DataFrame
    epochs_df = pd.DataFrame({
        'start': starts_time,
        'stop': stops_time,
        'peak_time': peaks_time,
        'peak_counts': peak_counts,
        'label': 'pbe',
    })
    
    epochs = Epoch(epochs=epochs_df, metadata=metadata)

    # Step 11: Merge nearby epochs if requested
    if sep is not None and sep > 0:
        epochs = epochs.merge_neighbors(max_epoch_sep=sep)
        print(f"{len(epochs)} epochs remaining after merging (sep={sep}s)")
    
    # Step 10: Apply duration filter
    if min_dur is not None or max_dur is not None:
        epochs = epochs.duration_slice(min_dur=min_dur, max_dur=max_dur)
        print(f"{len(epochs)} epochs remaining with durations within ({min_dur}, {max_dur})")
    
    
    return epochs


def detect_lowstates_epochs():
    pass
