import numpy as np
import mne
from pathlib import Path
import warnings
from ..core import DataWriter

class Signal(DataWriter):
    """
    Signal class for handling multi-channel time-series data with MNE integration.
    
    This class wraps neural signal data (EEG, LFP, etc.) and provides an interface
    through MNE's Raw object for signal processing operations like filtering, 
    re-referencing, and visualization.
    
    Parameters
    ----------
    traces : np.ndarray
        Signal data array of shape (n_channels, n_frames) or (n_frames,)
    sampling_rate : float
        Sampling rate in Hz
    t_start : float, optional
        Start time in seconds (default: 0.0)
    channel_id : array-like, optional
        Channel identifiers. If None, uses sequential integers (default: None)
    source_file : str or Path, optional
        Path to source file (default: None)
    metadata : dict, optional
        Additional metadata (default: None)
    ch_types : str, optional
        MNE channel type (default: 'seeg')
    
    Attributes
    ----------
    traces : np.ndarray
        Signal traces (retrieved from MNE object)
    mne : mne.io.RawArray
        MNE Raw object for signal processing
    channel_id : np.ndarray
        Channel identifiers
    t_start : float
        Start time in seconds
    sampling_rate : float
        Sampling rate in Hz
    
    Examples
    --------
    >>> traces = np.random.randn(64, 10000)  # 64 channels, 10000 samples
    >>> sig = Signal(traces, sampling_rate=1250.0)
    >>> sig.mne.filter(1, 100)  # Apply bandpass filter using MNE
    >>> filtered_traces = sig.traces  # Get filtered data
    
    Notes
    -----
    - All trace data is stored and accessed through the MNE object
    - Use `update_traces()` to sync the traces attribute after MNE operations
    - MNE operations are performed in-place on the mne attribute
    """
    
    def __init__(
        self,
        traces,
        sampling_rate,
        t_start=0.0,
        channel_id=None,
        source_file=None,
        metadata=None,
        ch_types='seeg',
    ) -> None:
        # Initialize parent DataWriter class with metadata
        super().__init__(metadata=metadata)
        
        # Ensure traces is at most 2D (channels × frames)
        assert traces.ndim <= 2
        
        # Convert 1D traces to 2D by adding channel dimension
        self.traces = traces if traces.ndim == 2 else traces[None, :]
        
        # Store temporal and sampling information
        self.t_start = t_start
        self._sampling_rate = sampling_rate
        
        # Set channel IDs - use sequential integers if not provided
        if channel_id is None:
            self.channel_id = np.arange(self.n_channels)
        else:
            self.channel_id = channel_id
        
        # Store source file path if provided
        self.source_file = source_file
        
        # Create MNE Raw object for signal processing capabilities
        # Convert channel IDs to strings for MNE compatibility
        ch_names = [f'{ch_id}' for ch_id in self.channel_id]
        info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types=ch_types)
        self.mne = mne.io.RawArray(self.traces.astype(float), info, verbose=False)
        
        # Associate source file with MNE object if provided
        if source_file is not None:
            self.mne._filenames = [Path(source_file)]
    
    def update(self):
        """
        Update traces and sampling rate from MNE object.
        
        This method synchronizes the Signal object's data with any modifications
        made to the underlying MNE Raw object (e.g., after filtering, resampling,
        or other MNE operations).
        
        Updates:
        - self.traces: Signal data from mne.get_data()
        - self._sampling_rate: Sampling rate from mne.info['sfreq']
        - self.channel_id: Channel identifiers from mne channel names
        """
        # Retrieve current data from MNE object (may have been modified)
        self.traces = self.mne.get_data()
        
        # Update sampling rate in case of resampling
        self._sampling_rate = self.mne.info['sfreq']
        
        # Update channel IDs, converting numeric strings back to integers
        self.channel_id = np.array([int(ch) if ch.isdigit() else ch for ch in self.mne.ch_names])

    @property
    def t_stop(self):
        """Calculate end time based on start time and duration."""
        return self.t_start + self.duration
    
    @property
    def duration(self):
        """Calculate signal duration in seconds from frames and sampling rate."""
        return self.traces.shape[1] / self.sampling_rate
    
    @property
    def n_channels(self):
        """Return number of channels (first dimension of traces)."""
        return self.traces.shape[0]
    
    @property
    def n_frames(self):
        """Return number of time samples (last dimension of traces)."""
        return self.traces.shape[-1]
    
    @property
    def sampling_rate(self):
        """Get current sampling rate."""
        return self._sampling_rate
    
    @sampling_rate.setter
    def sampling_rate(self, srate):
        """Set new sampling rate (does not resample data)."""
        self._sampling_rate = srate
    
    @property
    def time(self):
        """
        Generate time vector for the signal.
        
        Returns array of time points from t_start to t_stop with endpoint=False
        to match the actual sample times and MNE .times (samples occur at start of each interval).
        """
        return np.linspace(self.t_start, self.t_stop, self.n_frames, endpoint=False)
    
    def get_traces(self, channel_id=None):
        """
        Get traces from MNE object.
        
        Parameters
        ----------
        channel_id : int, list, or None
            Specific channel(s) to retrieve. If None, returns all channels
            
        Returns
        -------
        np.ndarray
            Signal traces for requested channels
        """
        if channel_id is None:
            # Return all channels
            return self.mne.get_data()
        else:
            # Convert single channel to list for consistent handling
            if isinstance(channel_id, int):
                channel_id = [channel_id]
            # Find channel indices in the channel list
            ch_indices = [list(self.channel_id).index(ch) for ch in channel_id]
            # Return data for specific channels
            return self.mne.get_data(picks=ch_indices)
    
    def time_slice(self, channel_id=None, t_start=None, t_stop=None):
        """
        Extract a temporal slice of the signal with optional channel selection.
        
        Parameters
        ----------
        channel_id : int, list, or None
            Channel(s) to extract. If None, all channels are included
        t_start : float or None
            Start time in seconds. If None, uses signal's start time
        t_stop : float or None
            Stop time in seconds. If None, uses last time point from time property
        
        Returns
        -------
        Signal
            New Signal object containing the sliced data
        """
        # Convert single channel ID to list for consistent handling
        if isinstance(channel_id, int):
            channel_id = [channel_id]
        
        # Set default start time if not provided
        if t_start is None:
            t_start = self.t_start

        # Clip t_start if it's before signal start and warn user
        if t_start < self.t_start:
            warnings.warn(f"t_start ({t_start}) is before signal start ({self.t_start}). "
                        f"Cropping to {self.t_start}", UserWarning)
            t_start = self.t_start

        # Set default stop time using last time point (accounts for endpoint=False in time property)
        if t_stop is None:
            t_stop = self.time[-1]  # Use the last time point from the time property

        # Clip t_stop if it exceeds signal end and warn user
        if t_stop > self.time[-1]:
            warnings.warn(f"t_stop ({t_stop}) exceeds signal end ({self.time[-1]}). "
                        f"Cropping to {self.time[-1]}", UserWarning)
            t_stop = self.time[-1]
            
        # Ensure logical consistency: stop time must be after start time
        assert t_stop > t_start, f"t_stop ({t_stop}) should be greater than t_start ({t_start})"
        
        # Prepare channel selection for MNE operations
        if channel_id is None:
            # No channel selection - use all channels
            picks = None
            use_channel_id = self.channel_id
        else:
            # Find indices of requested channels in the channel list
            ch_indices = [list(self.channel_id).index(ch) for ch in channel_id]
            picks = ch_indices
            use_channel_id = channel_id
        
        # Convert absolute times to relative times for MNE's crop function
        # MNE expects times relative to the start of the recording
        tmin = t_start - self.t_start  # Time relative to signal start
        tmax = t_stop - self.t_start   # Time relative to signal start
        
        # Clip tmax if it exceeds MNE's time bounds and warn user
        mne_max_time = self.mne.times[-1]
        if tmax > mne_max_time:
            warnings.warn(f"tmax ({tmax}) exceeds MNE signal end ({mne_max_time}). "
                        f"Cropping to {mne_max_time}", UserWarning)
            tmax = mne_max_time
        
        # Create a copy of MNE object and crop to time window
        cropped_mne = self.mne.copy().crop(tmin=tmin, tmax=tmax)
        
        # Extract the actual trace data from the cropped MNE object
        if picks is not None:
            traces = cropped_mne.get_data(picks=picks)
        else:
            traces = cropped_mne.get_data()
        
        # Create and return a new Signal object with the sliced data
        return Signal(
            traces,
            self.sampling_rate,
            t_start,  # New signal starts at the slice start time
            use_channel_id,
            source_file=self.source_file,
        )
    
    def rescale(self, factor=0.95 * 1e-3):
        """
        Scales signal, use it for converting raw signal to volts.
        
        Parameters
        ----------
        factor : float
            Scaling factor to apply to all signal values
            Default is 0.95e-3 for typical raw to voltage conversion
            
        Returns
        -------
        Signal
            New Signal object with rescaled traces
        """
        # Get data from MNE and apply scaling factor
        rescaled_traces = self.mne.get_data() * factor
        
        # Create new Signal object with rescaled data
        return Signal(
            traces=rescaled_traces,
            sampling_rate=self.sampling_rate,
            t_start=self.t_start,
            channel_id=self.channel_id,
            source_file=self.source_file,
        )
    
    def copy(self):
        """Create a copy of the Signal object with independent MNE object."""
        return Signal(
            traces=self.traces.copy(),  # Copy numpy array
            sampling_rate=self.sampling_rate,
            t_start=self.t_start,
            channel_id=self.channel_id.copy(),
            source_file=self.source_file,
            metadata=self.metadata.copy() if self.metadata else None
        )
        
    def __repr__(self):
        """
        Return detailed representation of Signal object.
        
        Provides comprehensive information about the signal including
        duration, channels, sampling rate, memory usage, and metadata.
        """
        # Build list of information lines for display
        info_lines = [
            f"Signal object:",
            f"  Duration: {self.duration:.3f} s ({self.t_start:.3f} - {self.t_stop:.3f} s)",
            f"  Channels: {self.n_channels} " if self.n_channels > 0 else "  Channels: 0",
            f"  Sampling rate: {self.sampling_rate:.2f} Hz",
            f"  Samples: {self.n_frames:,} frames",
            f"  Shape: {self.traces.shape} (channels × frames)",
            f"  Data type: {self.traces.dtype}",
            f"  Memory: {self.traces.nbytes / 1024**2:.2f} MB",
        ]
        
        # Add channel type information from MNE if available
        if hasattr(self, 'mne') and self.mne is not None:
            ch_types = set(self.mne.get_channel_types())
            info_lines.append(f"  Channel types: {', '.join(ch_types)}")
        
        # Include source file path if available
        if self.source_file is not None:
            info_lines.append(f"  Source: {self.source_file}")
        
        # Include metadata keys if metadata exists
        if hasattr(self, 'metadata') and self.metadata:
            info_lines.append(f"  Metadata keys: {', '.join(self.metadata.keys())}")
        
        # Join all lines into formatted string
        return '\n'.join(info_lines)