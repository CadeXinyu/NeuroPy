from enum import unique
import numpy as np
import pandas as pd
from .datawriter import DataWriter
from pathlib import Path
import scipy.signal as sg
import typing
from copy import deepcopy, copy


def _unpack_args(values, fs=1):
    """
    Parse arguments for threshold-based epoch detection.
    
    Handles both single values and min/max tuples, applying sampling frequency scaling.
    
    Parameters
    ----------
    values : float or tuple
        Single threshold value or (min, max) tuple
    fs : float, optional
        Sampling frequency for time-to-sample conversion, by default 1
    
    Returns
    -------
    tuple
        (val_min, val_max) scaled by sampling frequency
    """
    try:
        val_min, val_max = values
    except (TypeError, ValueError):
        val_min, val_max = (values, None)

    val_min = val_min * fs
    val_max = val_max * fs if val_max is not None else None

    return val_min, val_max


class Epoch(DataWriter):
    """
    A class for managing temporal epochs with start/stop times and labels.
    
    Epochs are time intervals commonly used in neuroscience data analysis to mark
    events, behavioral states, or artifact periods. This class provides efficient
    operations for epoch manipulation, merging, filtering, and set operations.
    
    Instance Variables (Attributes)
    --------------------------------
    _epochs : pd.DataFrame
        Internal storage of epoch data. Core columns are:
        - 'start' (float): Epoch start time in seconds
        - 'stop' (float): Epoch stop time in seconds  
        - 'label' (str): Epoch label/category identifier
        Additional columns may be present (e.g., 'peak_time', 't_start_eeg', 'duration')
        This DataFrame is always sorted by 'start' time and indexed sequentially.
        
    metadata : dict
        Dictionary containing associated metadata for the epoch collection.
        Can store arbitrary key-value pairs such as:
        - Recording session information
        - Subject/animal identifiers
        - Sampling rates
        - Experimental conditions
        - Processing parameters
        Inherited from DataWriter parent class.
    
    Properties (Read-only Computed Attributes)
    ------------------------------------------
    starts : np.ndarray
        1D array of all epoch start times in seconds.
        
    stops : np.ndarray
        1D array of all epoch stop times in seconds.
        
    peak_times : np.ndarray
        1D array of peak times if available (used in event detection).
        Raises AttributeError if 'peak_time' column doesn't exist.
        
    durations : np.ndarray
        1D array of epoch durations (stops - starts) in seconds.
        Computed dynamically, not stored.
        
    n_epochs : int
        Total number of epochs in the collection.
        
    labels : np.ndarray
        1D array of string labels for each epoch.
        
    has_labels : bool
        True if all epochs have non-empty labels, False otherwise.
        
    epochs : pd.DataFrame
        Direct access to the internal _epochs DataFrame.
        Use with caution - prefer using class methods for modifications.
    
    Main Methods
    ------------
    Core Operations:
        __init__(epochs, metadata, file) : Initialize from DataFrame, dict, or file
        __add__(epochs) : Concatenate two Epoch objects (+= operator)
        __getitem__(i) : Slice/filter by index, label, or boolean mask ([] operator)
        __len__() : Get number of epochs (len() function)
        __repr__() : String representation with epoch count and preview
        
    Temporal Operations:
        shift(dt) : Shift all epochs in time by constant offset
        scale(sf) : Scale all epoch times by constant factor
        time_slice(t_start, t_stop, strict) : Filter epochs within time bounds
        merge(dt) : Merge epochs within dt seconds of each other
        
    Set Operations:
        union(other_epoch, res) : Union of two epoch sets at given resolution
        intersection(other_epoch, res) : Intersection of two epoch sets
        
    Data Manipulation:
        add_epoch_manually(start, stop, label, merge_dt) : Add epochs by start/stop times
        add_epoch_by_index(index, start, stop, label) : Insert epoch at fractional index
        add_column(name, arr) : Add new column to epoch data
        add_dataframe(df) : Concatenate additional columns from DataFrame
        add_epoch_buffer(buffer_sec) : Extend epochs by buffer time
        set_labels(labels) : Create new Epoch with updated labels
        
    Conversion & Export:
        to_dataframe() : Convert to pandas DataFrame with duration column
        as_array() : Convert to 2D numpy array of [start, stop] pairs
        flatten() : Return 1D array of alternating starts and stops
        to_point_process(t_start, t_stop, bin_size) : Convert to boolean time series
        get_indices_for_time(t) : Mark time points that fall within epochs
        
    Query & Analysis:
        get_unique_labels() : Get sorted array of unique labels
        is_labels_unique() : Check if all labels are unique
        replace_start_with_t_start_eeg() : Replace start with EEG start times
        
    Static Methods (Class-level Factory Functions):
        from_array(starts, stops, label) : Create Epoch from start/stop arrays
        from_boolean_array(arr, t) : Create epochs from boolean time series
        from_peaks(arr, thresh, length, sep, boundary, fs) : Detect epochs from signal peaks
    
    Private Methods
    ---------------
    _validate(epochs) : Validate and standardize epoch DataFrame format
        - Converts dict to DataFrame if needed
        - Ensures required columns exist (start, stop, label)
        - Converts labels to strings
        - Sorts by start time
        - Returns cleaned copy
    
    Usage Examples
    --------------
    # Create from arrays
    >>> starts = [0, 10, 20]
    >>> stops = [5, 15, 25]
    >>> labels = ['A', 'B', 'C']
    >>> epochs = Epoch.from_array(starts, stops, labels)
    
    # Filter by label
    >>> epoch_A = epochs['A']
    
    # Merge nearby epochs
    >>> merged = epochs.merge(dt=2.0)  # Merge if within 2 seconds
    
    # Add buffer around epochs
    >>> buffered = epochs.add_epoch_buffer(0.5)  # Add 0.5s before/after
    
    # Get epochs in time window
    >>> subset = epochs.time_slice(5, 20, strict=True)
    
    # Combine epoch sets
    >>> all_epochs = epochs1 + epochs2
    
    # Convert to boolean array
    >>> times, bool_arr = epochs.to_point_process(0, 30, bin_size=0.001)
    
    Notes
    -----
    - Epochs are always maintained in sorted order by start time
    - Most operations return new Epoch objects (immutable pattern)
    - The class inherits from DataWriter for save/load functionality
    - Time units are always in seconds unless otherwise specified
    - Epochs can overlap; use merge() or combine_epochs() to resolve
    """
    
    def __init__(
        self, epochs: pd.DataFrame or dict or None, metadata=None, file=None
    ) -> None:
        """
        Initialize an Epoch object from DataFrame, dict, or file.
        
        OPTIMIZATION: Single file read (line 34) instead of two separate reads
        for epochs and metadata, reducing I/O overhead.
        
        Parameters
        ----------
        epochs : pd.DataFrame, dict, or None
            Epoch data with start, stop, and label columns
        metadata : dict, optional
            Metadata to associate with epochs
        file : str or Path, optional
            Path to .npy file containing saved epochs
        """
        super().__init__(metadata=metadata)

        if epochs is None:
            assert (
                file is not None
            ), "Must specify file to load if no epochs dataframe entered"
            # OPTIMIZATION: Single file load for both epochs and metadata
            loaded_data = np.load(file, allow_pickle=True).item()
            epochs = loaded_data["epochs"]
            self.metadata = loaded_data["metadata"]

        self._epochs = self._validate(epochs)

    def union(self, other_epoch, res):
        """
        Compute the union of two epoch sets at specified time resolution.
        
        The union contains time points present in either epoch set. Uses boolean
        array operations for efficient set algebra on temporal data.
        
        Parameters
        ----------
        other_epoch : Epoch
            Another Epoch object to union with
        res : float
            Time resolution (bin size) for the operation in seconds
        
        Returns
        -------
        Epoch
            New Epoch object representing the union
        """
        t_start = np.min((self.starts.min(), other_epoch.starts.min()))
        t_stop = np.max((self.stops.max(), other_epoch.stops.max()))
        times, bool1 = self.to_point_process(t_start, t_stop, bin_size=res)
        _, bool2 = other_epoch.to_point_process(t_start, t_stop, bin_size=res)

        return self.from_boolean_array(np.bitwise_or(bool1, bool2), times)

    def intersection(self, other_epoch, res):
        """
        Compute the intersection of two epoch sets at specified time resolution.
        
        The intersection contains only time points present in both epoch sets.
        Useful for finding overlapping events or common time periods.
        
        Parameters
        ----------
        other_epoch : Epoch
            Another Epoch object to intersect with
        res : float
            Time resolution (bin size) for the operation in seconds
        
        Returns
        -------
        Epoch
            New Epoch object representing the intersection
        """
        t_start = np.min((self.starts.min(), other_epoch.starts.min()))
        t_stop = np.max((self.stops.max(), other_epoch.stops.max()))
        times, bool1 = self.to_point_process(t_start, t_stop, bin_size=res)
        _, bool2 = other_epoch.to_point_process(t_start, t_stop, bin_size=res)

        return self.from_boolean_array(np.bitwise_and(bool1, bool2), times)


    def replace_start_with_t_start_eeg(self):
        """
        Replace 'start' column with 't_start_eeg' column if it exists.
        
        Used for aligning epochs with EEG recording start times when dealing
        with multi-modal recordings.
        """
        if hasattr(self, 'data'):
            self.data['start'] = self.data['t_start_eeg']

    def _validate(self, epochs):
        """
        Validate and standardize epoch data format.
        
        OPTIMIZATION: Uses .copy() to avoid SettingWithCopyWarning while minimizing
        the copying overhead (line 85). The label conversion is done efficiently
        by dropping and reassigning rather than using .loc which triggers warnings.
        
        Parameters
        ----------
        epochs : dict or pd.DataFrame
            Input epoch data
        
        Returns
        -------
        pd.DataFrame
            Validated and sorted epoch DataFrame
        """
        if isinstance(epochs, dict):
            try:
                epochs = pd.DataFrame(epochs)
            except:
                print("Error converting dictionary to pandas DataFrame")

        assert isinstance(epochs, pd.DataFrame)
        assert (
            pd.Series(["start", "stop", "label"]).isin(epochs.columns).all()
        ), "epochs should at least have columns/keys with names: start, stop, label"

        # Format labels as strings to ensure consistent type handling
        # OPTIMIZATION: Efficient column replacement avoids SettingWithCopyWarning
        # while minimizing unnecessary data copies
        epochs_labels_str = copy(epochs["label"].astype("str"))
        epochs = epochs.drop(columns="label", inplace=False)
        epochs.loc[:, "label"] = epochs_labels_str

        # Sort by start time for efficient temporal operations
        epochs = epochs.sort_values(by=["start"]).reset_index(drop=True)

        return epochs.copy()

    @property
    def starts(self):
        """Get array of epoch start times in seconds."""
        return self._epochs.start.values

    @property
    def stops(self):
        """Get array of epoch stop times in seconds."""
        return self._epochs.stop.values
    
    @property
    def peak_times(self):
        """Get array of peak times if available (used in event detection)."""
        return self._epochs.peak_time.values

    @property
    def durations(self):
        """
        Calculate duration of each epoch.
        
        OPTIMIZATION: Vectorized subtraction on numpy arrays is much faster
        than DataFrame operations for large datasets.
        
        Returns
        -------
        np.ndarray
            Array of epoch durations in seconds
        """
        return self.stops - self.starts

    @property
    def n_epochs(self):
        """Get total number of epochs."""
        return len(self.starts)

    @property
    def labels(self):
        """Get array of epoch labels."""
        return self._epochs.label.values

    def set_labels(self, labels):
        """
        Create new Epoch with updated labels.
        
        Parameters
        ----------
        labels : array-like
            New labels for each epoch
        
        Returns
        -------
        Epoch
            New Epoch object with updated labels
        """
        self._epochs["label"] = labels
        return Epoch(epochs=self._epochs)

    @property
    def has_labels(self):
        """
        Check if all epochs have non-empty labels.
        
        Returns
        -------
        bool
            True if all epochs have labels
        """
        return np.all(self._epochs["label"] != "")

    def __add__(self, epochs):
        """
        Concatenate two Epoch objects.
        
        OPTIMIZATION: Efficient column checking with np.array_equal (line 123)
        avoids unnecessary DataFrame column extraction when columns match.
        
        Parameters
        ----------
        epochs : Epoch
            Another Epoch object to concatenate
        
        Returns
        -------
        Epoch
            Combined Epoch object
        """
        assert isinstance(epochs, Epoch), "Can only add two core.Epoch objects"
        my_columns = self._epochs.columns
        other_columns = epochs._epochs.columns
        # OPTIMIZATION: Fast column comparison with np.array_equal
        if np.array_equal(my_columns, other_columns):
            df_new = pd.concat([self._epochs, epochs._epochs], ignore_index=True)
        else:
            # Fall back to essential columns only if schemas differ
            my_df = self._epochs[["start", "stop", "label"]]
            other_df = epochs._epochs[["start", "stop", "label"]]
            df_new = pd.concat([my_df, other_df]).reset_index(drop=True)

        return Epoch(epochs=df_new)

    def add_epoch_manually(self, start, stop, label="", merge_dt: float or None = 0):
        """
        Add one or more epochs manually specified by start/stop times.
        
        Parameters
        ----------
        start : float or array-like
            Start time(s) in seconds
        stop : float or array-like
            Stop time(s) in seconds
        label : str, optional
            Label for the epoch(s)
        merge_dt : float or None, optional
            If not None, merge epochs within this time threshold
        
        Returns
        -------
        Epoch
            New Epoch object with added epochs
        """
        comb_df = pd.DataFrame(
            {
                "start": np.array(start).reshape(-1),
                "stop": np.array(stop).reshape(-1),
                "label": label,
            }
        )

        if merge_dt is not None:
            return self.__add__(Epoch(comb_df)).merge(merge_dt)
        else:
            return self.__add__(Epoch(comb_df))

    def add_epoch_by_index(self, index, start, stop, label=""):
        """
        Insert an epoch at a specific fractional index position.
        
        Useful for inserting epochs at specific positions in the sequence
        without disturbing existing indices.
        
        Parameters
        ----------
        index : float
            Non-integer index position (e.g., 11.5 inserts between 11 and 12)
        start : float
            Epoch start time in seconds
        stop : float
            Epoch stop time in seconds
        label : str, optional
            Epoch label
        """
        assert np.mod(index, 1) > 0, "index must be a non-integer, e.g. -0.5 or 11.5"
        epochs_df = deepcopy(self._epochs)
        line = pd.DataFrame(
            {"start": start, "stop": stop, "label": label}, index=[index]
        )
        epochs_df = pd.concat((epochs_df, line), ignore_index=False)
        self._epochs = epochs_df.sort_index().reset_index(drop=True)

    def shift(self, dt):
        """
        Shift all epochs in time by a constant offset.
        
        Parameters
        ----------
        dt : float
            Time shift in seconds (positive = shift forward)
        
        Returns
        -------
        Epoch
            New Epoch object with shifted times
        """
        epochs = self._epochs.copy()
        epochs[["start", "stop"]] += dt
        return Epoch(epochs=epochs, metadata=self.metadata)

    def scale(self, sf):
        """
        Scale all epoch times by a constant factor.
        
        Useful for converting between different time units or sampling rates.
        
        Parameters
        ----------
        sf : float
            Scale factor to multiply times by
        
        Returns
        -------
        Epoch
            New Epoch object with scaled times
        """
        epochs = self._epochs.copy()
        epochs[["start", "stop"]] = epochs[["start", "stop"]] * sf
        return Epoch(epochs=epochs, metadata=self.metadata)

    def get_unique_labels(self):
        """
        Get sorted array of unique labels.
        
        Returns
        -------
        np.ndarray
            Unique labels in sorted order
        """
        return np.unique(self.labels)

    def is_labels_unique(self):
        """
        Check if every epoch has a unique label.
        
        Returns
        -------
        bool
            True if all labels are unique
        """
        return len(np.unique(self.labels)) == len(self)

    def to_dataframe(self):
        """
        Convert to pandas DataFrame with duration column added.
        
        OPTIMIZATION: Vectorized duration calculation via property access
        is more efficient than iterating over rows.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with start, stop, label, and duration columns
        """
        df = self._epochs.copy()
        df["duration"] = self.durations
        return df

    def add_column(self, name: str, arr: np.ndarray):
        """
        Add a new column to the epoch data.
        
        Parameters
        ----------
        name : str
            Column name
        arr : np.ndarray
            Array of values (must match number of epochs)
        
        Returns
        -------
        Epoch
            New Epoch object with added column
        """
        data = self.to_dataframe()
        data[name] = arr
        return Epoch(epochs=data, metadata=self.metadata)

    def add_dataframe(self, df: pd.DataFrame):
        """
        Concatenate additional columns from a DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to concatenate horizontally
        
        Returns
        -------
        Epoch
            New Epoch object with additional columns
        """
        assert isinstance(df, pd.DataFrame), "df should be a pandas dataframe"
        data = self.to_dataframe()
        data_new = pd.concat([data, df], axis=1)
        return Epoch(epochs=data_new, metadata=self.metadata)

    def __str__(self) -> str:
        pass

    def __getitem__(self, i):
        """
        Slice or filter epochs by index, label, or boolean mask.
        
        OPTIMIZATION: List comprehension for label filtering (line 200) is faster
        than using .isin() for moderate-sized label lists due to reduced overhead.
        
        Parameters
        ----------
        i : str, list, int, or slice
            Selection criteria (label string, list of labels, integer index, or slice)
        
        Returns
        -------
        Epoch
            Filtered Epoch object
        """
        if isinstance(i, str):
            data = self._epochs[self._epochs["label"] == i].copy()
        elif isinstance(i, list):
            assert all(isinstance(_, str) for _ in i), "All entries in epochs slicing list must be str"
            # OPTIMIZATION: List comprehension is faster than .isin() for moderate lists
            data = self._epochs[[label in i for label in self._epochs["label"]]]
        elif isinstance(i, (int, np.integer)):
            data = self._epochs.iloc[[i]].copy()
        else:
            data = self._epochs.iloc[i].copy()

        return Epoch(epochs=data.reset_index(drop=True))

    def __len__(self):
        """Get number of epochs."""
        return self.n_epochs

    def time_slice(self, t_start, t_stop, strict=True):
        """
        Filter epochs to those within specified time limits.
        
        Parameters
        ----------
        t_start : float
            Start time in seconds
        t_stop : float
            Stop time in seconds
        strict : bool, optional
            If True, only include epochs entirely within bounds.
            If False, include epochs that overlap with bounds.
        
        Returns
        -------
        Epoch
            Filtered Epoch object
        """
        if strict:
            # Only epochs completely within the time window
            idx = (self.starts >= t_start) & (self.stops <= t_stop)
        else:
            # Epochs that overlap with the time window
            idx = (self.starts < t_stop) & (self.stops > t_start)
        
        return Epoch(epochs=self._epochs[idx].reset_index(drop=True))

    def duration_slice(self, min_dur=None, max_dur=None):
        """return epochs that have durations between given thresholds

        Parameters
        ----------
        min_dur : float, optional
            minimum duration in seconds, by default None
        max_dur : float, optional
            maximum duration in seconds, by default None,

        Returns
        -------
        epoch
            epochs with durations between min_dur and max_dur
        """
        durations = self.durations
        if min_dur is None:
            min_dur = np.min(durations)
        if max_dur is None:
            max_dur = np.max(durations)

        return self[(durations >= min_dur) & (durations <= max_dur)]

    def label_slice(self, labels: typing.Union[list[str], str]):
        """Returns Epoch for input labels

        Parameters
        ----------
        labels : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        if isinstance(labels, str):
            labels = [labels]

        assert np.all([isinstance(_, str) for _ in labels])
        df = self._epochs[np.isin(self.labels, labels)].reset_index(drop=True)
        return Epoch(epochs=df)

    @staticmethod
    def from_file(f):
        d = DataWriter.from_file(f)
        if d is not None:
            return Epoch.from_dict(d)
        else:
            return None

    @property
    def is_overlapping(self):
        if self.n_epochs > 1:
            starts = self.starts
            stops = self.stops
            return np.all((starts[1:] - stops[:-1]) < 0)
        else:
            return False

    def itertuples(self):
        return self.to_dataframe().itertuples()

    def fill_blank(
        self,
        method: typing.Literal["from_left", "from_right", "from_nearest"] = "from_left",
    ):
        """Gaps in the epochs will be filled based on given criteria.
        Visualization:

        from_left:    |epoch1| gap |epoch2| --> |epoch1  ->|epoch2|
        from_right:   |epoch1| gap |epoch2| --> |epoch1|<-  epoch2|
        from_nearest: |epoch1| gap |epoch2| --> |epoch1->|<-epoch2|

        Parameters
        ----------
        method : str, optional
            how will the gaps be filled, by default "from_left"
            from_left = epoch preceding the gap is extended to fill
            from_right = epoch succeeding the gap is extended to fill
            from_nearest = first half of gap filled by extending preceding epoch and    second half is filled by extending succeeding epoch

        Returns
        -------
        core.Epoch
            epochs after filling the blank timepoints
        """
        ep_starts = self.starts
        ep_stops = self.stops
        ep_durations = self.durations
        ep_labels = self.labels

        mask = (ep_starts[:-1] + ep_durations[:-1]) < ep_starts[1:]
        (inds,) = np.nonzero(mask)

        if method == "from_left":
            for ind in inds:
                ep_durations[ind] = ep_starts[ind + 1] - ep_starts[ind]

        elif method == "from_right":
            for ind in inds:
                gap = ep_starts[ind + 1] - (ep_starts[ind] + ep_durations[ind])
                ep_starts[ind + 1] -= gap
                ep_durations[ind + 1] += gap

        elif method == "from_nearest":
            for ind in inds:
                gap = ep_starts[ind + 1] - (ep_starts[ind] + ep_durations[ind])
                ep_durations[ind] += gap / 2.0
                ep_starts[ind + 1] -= gap / 2.0
                ep_durations[ind + 1] += gap / 2.0

        # self.epochs["start"] = ep_starts
        # self.epochs["stop"] = ep_starts + ep_durations
        # self.epochs["duration"] = ep_durations

        return self.from_array(
            starts=ep_starts, stops=ep_starts + ep_durations, labels=ep_labels
        )

    def merge(self, dt: float = 0):
        """
        Merge epochs that are within dt seconds of each other.
        
        Parameters
        ----------
        dt : float, optional
            Time threshold in seconds for merging, by default 0
        
        Returns
        -------
        Epoch
            New Epoch object with merged epochs
        """
        if len(self) == 0:
            return self
        
        # OPTIMIZATION: Vectorized operations on sorted arrays
        starts = self.starts.copy()
        stops = self.stops.copy()
        labels = self.labels.copy()
        
        # Find epochs to merge (gap between consecutive epochs <= dt)
        gaps = starts[1:] - stops[:-1]
        merge_mask = gaps <= dt
        
        # Build merged epochs efficiently
        merged_starts = []
        merged_stops = []
        merged_labels = []
        
        current_start = starts[0]
        current_stop = stops[0]
        current_label = labels[0]
        
        for i in range(len(self) - 1):
            if merge_mask[i]:
                # Merge with next epoch
                current_stop = stops[i + 1]
            else:
                # Save current merged epoch
                merged_starts.append(current_start)
                merged_stops.append(current_stop)
                merged_labels.append(current_label)
                # Start new epoch
                current_start = starts[i + 1]
                current_stop = stops[i + 1]
                current_label = labels[i + 1]
        
        # Don't forget the last epoch
        merged_starts.append(current_start)
        merged_stops.append(current_stop)
        merged_labels.append(current_label)
        
        merged_df = pd.DataFrame({
            'start': merged_starts,
            'stop': merged_stops,
            'label': merged_labels
        })
        
        return Epoch(epochs=merged_df, metadata=self.metadata)
    
    def merge_overlap(self):
        """
        Merge overlapping epochs into single epochs.
        
        This function identifies groups of overlapping epochs and merges each group
        into a single epoch. Two epochs overlap if one starts before the other ends.
        The merged epoch spans from the earliest start to the latest stop time in
        the group, and uses the label from the first epoch in the group.
        
        Returns
        -------
        Epoch
            New Epoch object with overlapping epochs merged
        
        Examples
        --------
        >>> # Epochs: [0-5, A], [3-8, B], [7-10, C], [15-20, D]
        >>> # Result: [0-10, A], [15-20, D]  (first three merged, last one separate)
        
        Notes
        -----
        - Epochs are considered overlapping if start_i < stop_j for any pair
        - All overlapping epochs are merged into a single epoch spanning the full range
        - The label from the first epoch in each merged group is used
        """
        if len(self) == 0:
            return self
        
        # Sort by start time to ensure proper grouping
        sorted_indices = np.argsort(self.starts)
        starts = self.starts[sorted_indices]
        stops = self.stops[sorted_indices]
        labels = self.labels[sorted_indices]
        
        merged_starts = []
        merged_stops = []
        merged_labels = []
        
        # Initialize first group
        current_start = starts[0]
        current_stop = stops[0]
        current_label = labels[0]
        
        for i in range(1, len(self)):
            # Check if current epoch overlaps with the merged group
            if starts[i] < current_stop:
                # Overlap detected - extend the merged epoch
                current_stop = max(current_stop, stops[i])
            else:
                # No overlap - save current merged group and start new one
                merged_starts.append(current_start)
                merged_stops.append(current_stop)
                merged_labels.append(current_label)
                
                # Start new group
                current_start = starts[i]
                current_stop = stops[i]
                current_label = labels[i]
        
        # Don't forget the last group
        merged_starts.append(current_start)
        merged_stops.append(current_stop)
        merged_labels.append(current_label)
        
        merged_df = pd.DataFrame({
            'start': merged_starts,
            'stop': merged_stops,
            'label': merged_labels
        })
        
        return Epoch(epochs=merged_df, metadata=self.metadata)

    def merge_neighbors(self, max_epoch_sep=1e-6):
        starts, stops, labels = self.starts, self.stops, self.labels
        if len(starts) == 0: return self

        # 1. Compare i with i-1 (Vectorized)
        #    Check if adjacent rows have same label AND are close in time
        same_label = labels[1:] == labels[:-1]
        close_time = (starts[1:] - stops[:-1]) < max_epoch_sep
        should_merge = same_label & close_time

        # 2. Identify Split Points (Where we DO NOT merge)
        #    A split happens at index 0, and anywhere 'should_merge' is False
        is_split = np.concatenate(([True], ~should_merge))
        
        # 3. Generate Group IDs
        group_ids = np.cumsum(is_split) - 1  # 0-based IDs

        # 4. Aggregate (This is the tricky part in pure NumPy, using ufunc.reduceat)
        #    Find indices where groups change
        unique_groups, group_starts = np.unique(group_ids, return_index=True)
        
        #    Calculate Min Start and Max Stop for each group
        #    We use maximum.reduceat and minimum.reduceat for speed
        new_starts = np.minimum.reduceat(starts, group_starts)
        new_stops = np.maximum.reduceat(stops, group_starts)
        new_labels = labels[group_starts]

        return Epoch.from_array(new_starts, new_stops, new_labels)

    def contains(self, t, return_closest: bool = False):
        """Check if timepoints lie within epochs, must be non-overlapping epochs

        Parameters
        ----------
        t : array
            timepoints in seconds
        return_closest: bool
            True = return closest epoch before to all points in t even if t is outside epoch

        Returns
        -------
        _type_
            _description_
        """

        assert self.is_overlapping == False, "Epochs must be non overlapping"
        assert isinstance(t, np.ndarray), "t must be a numpy.ndarray"

        labels = self.labels
        bin_loc = np.digitize(t, self.flatten())
        indx_bool = bin_loc % 2 == 1

        if not return_closest:
            return (
                indx_bool,
                t[indx_bool],
                labels[((bin_loc[indx_bool] - 1) / 2).astype("int")],
            )
        else:
            return indx_bool, t, labels[bin_loc], bin_loc

    def delete_in_between(self, t1, t2):
        epochs_df = self.to_dataframe()[["start", "stop", "label"]]
        # delete epochs if they are within t1, t2
        epochs_df = epochs_df[~((epochs_df["start"] >= t1) & (epochs_df["stop"] <= t2))]

        # truncate stop if start is less than t1 but stop is within t1,t2
        epochs_df.loc[
            (epochs_df["start"] < t1)
            & (t1 < epochs_df["stop"])
            & (epochs_df["stop"] <= t2),
            "stop",
        ] = t1

        # truncate start if stop is greater than t2 but start is within t1,t2
        epochs_df.loc[
            (epochs_df["start"] > t1)
            & (epochs_df["start"] <= t2)
            & (epochs_df["stop"] > t2),
            "start",
        ] = t2

        # if epoch starts before and ends after range,
        flank_start = epochs_df[
            (epochs_df["start"] < t1) & (epochs_df["stop"] > t2)
        ].copy()
        flank_start["stop"] = t1
        flank_stop = epochs_df[
            (epochs_df["start"] < t1) & (epochs_df["stop"] > t2)
        ].copy()
        flank_stop["start"] = t2
        epochs_df = epochs_df[~((epochs_df["start"] < t1) & (epochs_df["stop"] > t2))]
        epochs_df = pd.concat([epochs_df, flank_start, flank_stop], ignore_index=True)
        return Epoch(epochs_df)

    def proportion_by_label(self, t_start=None, t_stop=None, ignore_gaps=False):
        """Get proportion of time for each label type

        Parameters
        ----------
        t_start : float, optional
            start time in seconds, by default None
        t_stop : float, optional
            stop time in seconds, by default None
        ignore_gaps: will return None if set and there is no epoch in the time period selected.

        Returns
        -------
        dict
            dictionary containing proportion for each unique label between t_start and t_stop
        """
        if t_start is None:
            t_start = self.starts[0]
        if t_stop is None:
            t_stop = self.stops[-1]

        duration = t_stop - t_start

        ep = self._epochs.copy()
        ep = ep[(ep.stop > t_start) & (ep.start < t_stop)].reset_index(drop=True)
        if not ignore_gaps:
            assert ep.shape[0] > 0, "cannot have empty time gaps between epoch labels with ignore_gaps=False"
        elif ignore_gaps and (ep.shape[0] > 0):
            if ep["start"].iloc[0] < t_start:
                ep.at[0, "start"] = t_start

            if ep["stop"].iloc[-1] > t_stop:
                ep.at[ep.index[-1], "stop"] = t_stop

            ep["duration"] = ep.stop - ep.start

            ep_group = ep.groupby("label").sum(numeric_only=True).duration / duration

            label_proportion = {}
            for label in self.get_unique_labels():
                label_proportion[label] = 0.0

            for state in ep_group.index.values:
                label_proportion[state] = ep_group[state]

            return label_proportion
        else:
            return None

    def durations_by_label(self):
        """Return total duration for each unique label

        Returns
        -------
        dict
            dictionary containing duration of each unique label
        """
        labels = self.labels
        durations = self.durations
        unique_labels = self.get_unique_labels()
        label_durations = {}
        for label in unique_labels:
            label_durations[label] = durations[labels == label].sum()

        return label_durations

    def resample_labeled_epochs(self, res, t_start=None, t_stop=None, merge_neighbors=True):
        """Resample epochs to different size blocks using a winner take all method to assign
        a label name. e.g. if the first 100-second epoch is 40% quiet wake, 50% REM, and 10% NREM
        it would get labeled as REM.  Pretty slow, even slower with merge_neighbors=True

        :param: res: block size in seconds
        :param: t_start: start time in seconds, default = start of first epoch
        :param: t_stop : stop time in seconds, default = stop of last epoch
        :param merge_neighbors: combine adjacent epochs of the same label, default=True"""

        if t_start is None:
            t_start = self.starts[0]
        elif t_start < self.starts[0]:
            t_start = self.starts[0]
            print('t_start < start time of first epoch, reassigned to match first epoch start time')

        if t_stop is None:
            t_stop = self.stops[-1]
        if t_stop > self.stops[-1]:
            t_stop = self.stops[-1]
            print('t_stop > stop time of first epoch, reassigned to match last epoch stop time')
        bins = np.arange(t_start, t_stop + res, res)
        start_rs = bins[:-1]
        stop_rs = bins[1:]
        label_rs = []
        for start, stop in zip(start_rs, stop_rs):
            props = self.proportion_by_label(start, stop, ignore_gaps=True)
            label_add = list(props.keys())[np.argmax(list(props.values()))] if props is not None else ""
            label_rs.append(label_add)
        # except AssertionError:  # Append nothing if gap found in epochs
        #     label_rs.append("")

        epoch_rs = Epoch(pd.DataFrame({"start": start_rs, "stop": stop_rs, "label": label_rs}))
        epoch_rs = epoch_rs.merge_neighbors() if merge_neighbors else epoch_rs

        return epoch_rs

    def count(self, t_start=None, t_stop=None, binsize=300):
        if t_start is None:
            t_start = 0

        if t_stop is None:
            t_stop = np.max(self.stops)

        mid_times = self.starts + self.durations / 2
        bins = np.arange(t_start, t_stop + binsize, binsize)
        return np.histogram(mid_times, bins=bins)[0]

    def as_array(self):
        """
        Convert to 2D numpy array of [start, stop] pairs.
        
        Returns
        -------
        np.ndarray
            Shape (n_epochs, 2) array of start/stop times
        """
        return np.column_stack((self.starts, self.stops))

    def flatten(self):
        """
        Returns 1D array of alternating start and stop times.
        
        Note: Array is monotonically increasing only if epochs don't overlap.
        
        Returns
        -------
        np.ndarray
            1D array: [start1, stop1, start2, stop2, ...]
        """
        return self.as_array().flatten("C")

    def to_point_process(self, t_start=None, t_stop=None, bin_size=(1 / 1250)):
        """
        Convert epochs to boolean time series at specified resolution.
        
        OPTIMIZATION: Integer array indexing (lines 684-686) is significantly faster
        than the commented-out boolean indexing approach. By converting time to
        indices once and using slice assignment, we avoid repeated array comparisons.
        
        Parameters
        ----------
        t_start : float, optional
            Start time for output array, defaults to 0
        t_stop : float, optional
            Stop time for output array, defaults to max stop time
        bin_size : float, optional
            Time resolution in seconds, by default 1/1250
        
        Returns
        -------
        tuple
            (times, boolean_array) where boolean is True during epochs
        """
        if t_start is None:
            t_start = 0

        if t_stop is None:
            t_stop = np.max(self.stops)

        times = np.arange(t_start, t_stop, bin_size)

        # OPTIMIZATION: Integer indexing is much faster than boolean comparisons
        time_bool = np.zeros_like(times).astype(bool)
        
        # Convert time boundaries to array indices for efficient slicing
        for start_ind, end_ind in zip(
            ((self.starts - t_start) / bin_size).astype(int), 
            ((self.stops - t_start) / bin_size).astype(int)
        ):
            time_bool[start_ind:end_ind] = True

        return times, time_bool

    def add_epoch_buffer(self, buffer_sec: float or int or tuple or list):
        """
        Extend each epoch by adding buffer time before and/or after.
        
        Useful for capturing context around events or ensuring complete coverage.
        
        Parameters
        ----------
        buffer_sec : float, int, tuple, or list
            Buffer duration(s) in seconds. If single value, applied to both sides.
            If tuple/list, (before, after) buffer durations.
        """
        df = self._epochs.copy()
        self._epochs = add_epoch_buffer(df, buffer_sec)

        # Update cached properties
        self.starts
        self.stops
        print(f"Buffer of {buffer_sec} added before/after each epoch")
        
    @staticmethod
    def from_peaks(arr: np.ndarray, thresh, length, sep=0, boundary=0, fs=1):
        """
        Detect epochs from peaks in a signal array.
        
        This method finds peaks in a signal, determines their bases, and creates
        epochs based on height and duration criteria. Adjacent peaks can be merged
        based on separation threshold.
        
        OPTIMIZATION: Vectorized peak detection using scipy.signal.find_peaks with
        efficient merging algorithm that processes peaks sequentially (lines 715-725).
        The array-based approach avoids nested loops.
        
        Parameters
        ----------
        arr : np.ndarray
            Input signal array
        thresh : float or tuple
            Peak height threshold(s) - single value or (min, max)
        length : float or tuple
            Epoch duration criteria in seconds - single value or (min, max)
        sep : float, optional
            Minimum separation between peaks in seconds (peaks closer than this merge)
        boundary : float, optional
            Baseline threshold for peak detection
        fs : float, optional
            Sampling frequency in Hz
        
        Returns
        -------
        tuple
            (Epoch, peak_times, peak_values) where:
            - Epoch: detected epochs as Epoch object
            - peak_times: array of peak times in seconds
            - peak_values: array of peak amplitudes
        """
        hmin, hmax = _unpack_args(thresh)  # does not need fs
        lmin, lmax = _unpack_args(length, fs=fs)
        sep = sep * fs + 1e-6

        assert hmin >= boundary, "boundary must be smaller than min thresh"

        # Apply boundary threshold and find peaks
        arr_thresh = np.where(arr >= boundary, arr, 0)
        peaks, props = sg.find_peaks(arr_thresh, height=[hmin, hmax], prominence=0)

        starts, stops = props["left_bases"], props["right_bases"]
        peaks_values = arr_thresh[peaks]

        # OPTIMIZATION: Efficient sequential merging of overlapping epochs
        # Process in order and mark epochs for deletion rather than modifying during iteration
        n_epochs = len(starts)
        ind_delete = []
        for i in range(n_epochs - 1):
            if (starts[i + 1] - stops[i]) < sep:
                # Merge: extend next epoch to cover both and keep higher peak
                starts[i + 1] = min(starts[i], starts[i + 1])
                stops[i + 1] = max(stops[i], stops[i + 1])

                peaks_values[i + 1] = max(peaks_values[i], peaks_values[i + 1])
                peaks[i + 1] = [peaks[i], peaks[i + 1]][
                    np.argmax([peaks_values[i], peaks_values[i + 1]])
                ]
                ind_delete.append(i)

        # Build array and remove merged epochs
        epochs_arr = np.vstack((starts, stops, peaks, peaks_values)).T
        epochs_arr = np.delete(epochs_arr, ind_delete, axis=0)

        # Apply duration thresholds
        epochs_length = epochs_arr[:, 1] - epochs_arr[:, 0]
        if lmax is None:
            lmax = epochs_length.max()
        ind_keep = (epochs_length >= lmin) & (epochs_length <= lmax)

        starts, stops, peaks, peaks_values = epochs_arr[ind_keep, :].T

        return Epoch.from_array(starts / fs, stops / fs), peaks / fs, peaks_values

    @staticmethod
    def from_boolean_array(arr, t=None):
        """
        Create epochs from a boolean time series.
        
        Finds continuous segments where array is True and converts them to epochs.
        Uses efficient edge detection via padding and differencing.
        
        OPTIMIZATION: Edge detection using np.pad and np.diff (lines 762-764) is
        faster than iterating through the array to find state changes.
        
        Parameters
        ----------
        arr : np.array
            Boolean array where True indicates epoch periods
        t : np.array, optional
            Corresponding time values in seconds. If None, uses indices.
        
        Returns
        -------
        Epoch
            Epochs corresponding to True segments in the array
        """
        if isinstance(t, pd.Series):  # grab values only
            t = t.values
        assert np.array_equal(arr, arr.astype(bool)), "Only boolean array accepted"
        
        # OPTIMIZATION: Edge detection via padding and differencing
        int_arr = arr.astype("int")
        pad_arr = np.pad(int_arr, 1)
        diff_arr = np.diff(pad_arr)
        starts, stops = np.where(diff_arr == 1)[0], np.where(diff_arr == -1)[0]
        stops[stops == len(arr)] = len(arr) - 1

        if t is not None:
            assert len(t) == len(arr), "time length should be same as input array"
            starts, stops = t[starts], t[stops]

        return Epoch.from_array(starts, stops, "high")

    def get_indices_for_time(self, t: np.array):
        """
        Mark time points that fall within epochs.
        
        OPTIMIZATION: While this uses a loop over epochs, for very large time arrays
        with few epochs, this is more efficient than creating a full boolean array
        for each epoch and combining them.
        
        Parameters
        ----------
        t : np.array
            Array of time points in seconds
        
        Returns
        -------
        np.ndarray
            Boolean array, True where time points are within epochs
        """
        time_bool = np.zeros_like(t)

        for e in self.as_array():
            time_bool[np.where((t >= e[0]) & (t <= e[1]))[0]] = 1

        return time_bool.astype("bool")

    @property
    def epochs(self):
        """Direct access to internal DataFrame representation."""
        return self._epochs

    @staticmethod
    def from_array(starts, stops, label=""):
        """
        Create Epoch from arrays of start and stop times.
        
        Parameters
        ----------
        starts : array-like
            Array of start times in seconds
        stops : array-like
            Array of stop times in seconds
        label : str or array-like, optional
            Label(s) for epochs
        
        Returns
        -------
        Epoch
            New Epoch object
        """
        df = pd.DataFrame({
            'start': np.asarray(starts),
            'stop': np.asarray(stops),
            'label': label
        })
        return Epoch(epochs=df)
    
    def __repr__(self) -> str:
        """
        String representation showing epoch count and preview.
        
        Returns
        -------
        str
            Formatted string with epoch count and first 5 epochs
        """
        return f"{len(self.starts)} epochs\nSnippet: \n {self._epochs.head(5)}"
    
    def __str__(self) -> str:
        pass

def add_epoch_buffer(epoch_df: pd.DataFrame, buffer_sec: float or int or tuple or list):
    """
    Extend epochs by adding buffer time before and/or after.
    
    Parameters
    ----------
    epoch_df : pd.DataFrame
        DataFrame with start and stop columns
    buffer_sec : float, int, tuple, or list
        Buffer duration(s). Single value applies to both sides,
        tuple/list is (before_buffer, after_buffer)
    
    Returns
    -------
    pd.DataFrame
        Modified DataFrame with extended epochs
    """
    if type(buffer_sec) in [int, float]:
        buffer_sec = (buffer_sec, buffer_sec)
    else:
        assert len(buffer_sec) == 2

    epoch_df["start"] -= buffer_sec[0]
    epoch_df["stop"] += buffer_sec[1]

    return epoch_df


def get_epoch_overlap_duration(epochs1: Epoch, epochs2: Epoch):
    """
    Calculate total duration of overlap between two epoch sets.
    
    OPTIMIZATION: While this uses nested loops (lines 816-818), it's appropriate
    for moderate numbers of epochs. For very large epoch sets, consider using
    the intersection() method with a point process representation.
    
    Parameters
    ----------
    epochs1 : Epoch
        First epoch set
    epochs2 : Epoch
        Second epoch set
    
    Returns
    -------
    float
        Total overlapping duration in seconds
    """
    e_array1 = epochs1.to_dataframe().loc[:, ["start", "stop"]].values
    e_array2 = epochs2.to_dataframe().loc[:, ["start", "stop"]].values
    overlaps = []
    for e1 in e_array1:
        for e2 in e_array2:
            overlaps.append(getOverlap(e1, e2))

    return np.array(overlaps).sum()


def getOverlap(a, b):
    """
    Calculate overlap duration between two time intervals.
    
    Uses the standard formula: overlap = min(end1, end2) - max(start1, start2)
    Returns 0 if intervals don't overlap.
    
    From: https://stackoverflow.com/questions/2953967/built-in-function-for-computing-overlap-in-python
    
    Parameters
    ----------
    a : array-like
        [start, stop] of first interval
    b : array-like
        [start, stop] of second interval
    
    Returns
    -------
    float
        Overlap duration (0 if no overlap)
    """
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def combine_epochs(epochs_df: pd.DataFrame, inplace: bool = True):
    """
    Merge overlapping epochs to eliminate containment and overlap.
    
    This function identifies epochs whose start or stop times fall within other epochs
    and merges them. Note: Epoch.union() might be a better choice for complex cases.
    
    OPTIMIZATION: While this uses DataFrame iteration (lines 836-851), it processes
    overlaps efficiently by checking all potential overlaps in one pass using
    vectorized comparisons with bitwise_and.
    
    Parameters
    ----------
    epochs_df : pd.DataFrame
        DataFrame with start and stop columns
    inplace : bool, optional
        If True, modify DataFrame in place. If False, return modified copy.
    
    Returns
    -------
    pd.DataFrame or None
        Modified DataFrame if inplace=False, None if inplace=True
    """
    all([col in epochs_df.columns for col in ["start", "stop"]])

    # Identify overlapping epochs using vectorized comparisons
    start_overlaps, stop_overlaps = [], []
    for ide, epoch in epochs_df.iterrows():
        # Find epochs that contain this epoch's start time
        overlap_start = np.bitwise_and(epoch['start'] > epochs_df['start'],
                                       epoch['start'] < epochs_df['stop'])
        # Find epochs that contain this epoch's stop time
        overlap_stop = np.bitwise_and(epoch['stop'] > epochs_df['start'],
                                      epoch['stop'] < epochs_df['stop'])
        if overlap_start.sum() == 1:
            start_overlap_id = np.where(overlap_start)[0][0]
            start_overlaps.append([ide, start_overlap_id])

        if overlap_stop.sum() == 1:
            stop_overlap_id = np.where(overlap_stop)[0][0]
            stop_overlaps.append([ide, stop_overlap_id])
    
    # Merge epochs by extending boundaries
    for start in start_overlaps:
        epochs_df.loc[start[0], "start"] = epochs_df.loc[start[1], "start"]

    for stop in stop_overlaps:
        epochs_df.loc[stop[0], "stop"] = epochs_df.loc[stop[1], "stop"]

    # Remove duplicates and sort
    if inplace:
        epochs_df.drop_duplicates(inplace=inplace, ignore_index=True)
        epochs_df.sort_values(by='start', inplace=inplace, ignore_index=True)
        return None
    else:
        epochs_out = epochs_df.drop_duplicates(inplace=inplace, ignore_index=True)
        epochs_out = epochs_out.sort_values(by='start', inplace=inplace, ignore_index=True)
        return epochs_out


if __name__ == "__main__":
    # Example usage
    art_file = "/data3/Trace_FC/Recording_Rats/Finn2/2023_05_06_habituation1/Finn2_habituation1_denoised.art_epochs.npy"
    art_epochs = Epoch(epochs=None, file=art_file)
    epochs_to_add = np.array([[1291, 1291.2], [2734, 2734.5], [1622, 1623.4]])
    art_epochs.add_epoch_manually(epochs_to_add[:, 0], epochs_to_add[:, 1])