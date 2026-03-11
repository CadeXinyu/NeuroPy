import numpy as np
import pandas as pd
from ..utils.mathutil import gaussian_kernel1D
import scipy.signal as sg
from .datawriter import DataWriter
from . import Epoch
from .. import core
from copy import deepcopy
from joblib import Parallel, delayed
from scipy import stats
from scipy.ndimage import gaussian_filter1d


class Neurons(DataWriter):
    """Class to hold a group of spiketrains and their labels, ids etc."""

    # TODO: Contemplate adding implicit support for noisy_epochs, such that firing_rate, get_binned_spiketrains, get_mua etc. deletes/ignores these time points for more accurate estimations

    def __init__(
        self,
        spiketrains: np.ndarray,
        t_stop,
        t_start=0.0,
        sampling_rate=1,
        neuron_ids=None,
        neuron_type=None,
        waveforms=None,
        waveforms_amplitude=None,
        peak_channels=None,
        clu_q=None,
        shank_ids=None,
        metadata=None,
    ) -> None:
        """Initializes the Neurons instance

        Parameters
        ----------
        spiketrains : np.array/list of numpy arrays
            each array contains spiketimes in seconds, 5 arrays for 5 neurons
        t_stop : float
            time when the recording was stopped
        t_start : float, optional
            start time for the recording/spike trains, by default 0.0
        sampling_rate : int, optional
            at what sampling rate the spike times were recorded, by default 1
        neuron_ids : array, optional
            id for each spiketrain/neuron, by default None
        neuron_type : array of strings, optional
            what neuron type, by default None
        waveforms : (n_neurons x n_channels x n_timepoints), optional
            waveshape for each neuron, by default None
        waveforms_amplitude : list/array of arrays, optional
            the number of arrays should match spiketrains, each value gives scaling factor used for template waveform to extract that spike, by default None
        peak_channels : array, optional
            peak channel for waveform, by default None
        shank_ids : array of int, optional
            which shank of the probe each spiketrain was recorded from, by default None
        metadata : dict, optional
            any additional metadata, by default None
        """
        super().__init__(metadata=metadata)

        self.spiketrains = np.array(spiketrains, dtype="object")
        if neuron_ids is None:
            self.neuron_ids = np.arange(len(self.spiketrains))
        else:
            self.neuron_ids = neuron_ids

        if waveforms is not None:
            assert (
                waveforms.shape[0] == self.n_neurons
            ), "Waveforms first dimension should match number of neurons"

        if waveforms_amplitude is not None:
            assert len(waveforms_amplitude) == len(
                self.spiketrains
            ), "length should match"
            self.waveforms_amplitude = waveforms_amplitude
        else:
            self.waveforms_amplitude = None

        self.waveforms = waveforms
        self.shank_ids = shank_ids
        self.neuron_type = neuron_type
        self.peak_channels = peak_channels
        self._sampling_rate = sampling_rate
        self.t_start = t_start
        self.t_stop = t_stop
        self.clu_q = clu_q

    @staticmethod
    def load(file):
        """Loads a previously saved Neurons class from an .npy file"""
        neurons_dict = DataWriter.from_file(file)

        return Neurons.from_dict(neurons_dict)

    def __getitem__(self, i):
        # copy object
        spiketrains = self.spiketrains[i]
        if self.neuron_type is not None:
            neuron_type = self.neuron_type[i]
        else:
            neuron_type = self.neuron_type

        if self.waveforms is not None:
            waveforms = self.waveforms[i]
        else:
            waveforms = self.waveforms

        if self.waveforms_amplitude is not None:
            waveforms_amplitude = self.waveforms_amplitude[i]
        else:
            waveforms_amplitude = self.waveforms_amplitude

        if self.peak_channels is not None:
            peak_channels = self.peak_channels[i]
        else:
            peak_channels = self.peak_channels

        if self.shank_ids is not None:
            shank_ids = self.shank_ids[i]
        else:
            shank_ids = self.shank_ids

        return Neurons(
            spiketrains=spiketrains,
            t_start=self.t_start,
            t_stop=self.t_stop,
            sampling_rate=self.sampling_rate,
            neuron_ids=self.neuron_ids[i],
            neuron_type=neuron_type,
            waveforms=waveforms,
            waveforms_amplitude=waveforms_amplitude,
            peak_channels=peak_channels,
            shank_ids=shank_ids,
        )

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @property
    def n_neurons(self):
        return len(self.spiketrains)

    def __repr__(self) -> str:
        try:
            neuron_types = np.unique(self.neuron_type)
        except TypeError:
            neuron_types = "Error importing - check inputs"
        return f"{self.__class__.__name__}\n n_neurons: {self.n_neurons}\n t_start: {self.t_start}\n t_stop: {self.t_stop}\n neuron_type: {neuron_types}"

    def time_slice(self, t_start=None, t_stop=None, zero_spike_times=False):
        """zero_spike_times = True will subtract t_start from all spike times"""
        t_start, t_stop = super()._time_slice_params(t_start, t_stop)
        neurons = deepcopy(self)
        if zero_spike_times:
            spiketrains = [t[(t >= t_start) & (t <= t_stop)] - t_start for t in neurons.spiketrains]
            t_stop = t_stop - t_start
            t_start = 0
        else:
            spiketrains = [t[(t >= t_start) & (t <= t_stop)] for t in neurons.spiketrains]

        return Neurons(
            spiketrains=spiketrains,
            t_stop=t_stop,
            t_start=t_start,
            sampling_rate=neurons.sampling_rate,
            neuron_ids=neurons.neuron_ids,
            neuron_type=neurons.neuron_type,
            waveforms=neurons.waveforms,
            peak_channels=neurons.peak_channels,
            shank_ids=neurons.shank_ids,
        )

    def neuron_slice(self, neuron_inds=None, neuron_ids=None):
        neurons = deepcopy(self)

        if neuron_inds is not None and neuron_ids is not None:
            raise ValueError("Specify either neuron_inds or neuron_ids, but not both.")

        # Handle selection of neuron indices
        if neuron_inds is not None:
            if isinstance(neuron_inds, int):
                neuron_inds = [neuron_inds]

            # Find the positional indices directly from the numpy array
            positions = np.array(neuron_inds)  # Use provided indices directly

        # Handle selection by neuron IDs (MODIFIED SECTION)
        elif neuron_ids is not None:
            if isinstance(neuron_ids, int):
                neuron_ids = [neuron_ids]

            # --- KEY CHANGE START ---
            # Create a lookup dictionary to map neuron_id -> original_index
            # This allows O(1) lookup
            id_to_idx_map = {uid: i for i, uid in enumerate(neurons.neuron_ids)}
            
            try:
                # Iterate through the INPUT neuron_ids to preserve their order
                positions = np.array([id_to_idx_map[uid] for uid in neuron_ids])
            except KeyError as e:
                # Raise error if a requested ID does not exist in the object
                raise ValueError(f"Neuron ID {e} not found in the current object.")
            # --- KEY CHANGE END ---

        else:
            raise ValueError("Must specify either neuron_inds or neuron_ids.")

        # Extract data using the found positions
        spiketrains = neurons.spiketrains[positions]
        neuron_type = None if neurons.neuron_type is None else neurons.neuron_type.iloc[positions]
        waveforms = None if neurons.waveforms is None else neurons.waveforms[positions]
        waveforms_amplitude = None if neurons.waveforms_amplitude is None else neurons.waveforms_amplitude[positions]
        peak_channels = None if neurons.peak_channels is None else neurons.peak_channels[positions]
        shank_ids = None if neurons.shank_ids is None else neurons.shank_ids[positions]
        clu_q = None if neurons.clu_q is None else neurons.clu_q[positions]

        # Ensure neuron_ids remains a NumPy array and retains its original values
        sliced_neuron_ids = np.array(neurons.neuron_ids)[positions]

        # Create and return the new Neurons object
        return Neurons(
            spiketrains=spiketrains,
            t_stop=neurons.t_stop,
            t_start=neurons.t_start,
            sampling_rate=neurons.sampling_rate,
            neuron_ids=sliced_neuron_ids,  # This will now match the input order
            neuron_type=neuron_type,
            waveforms=waveforms,
            waveforms_amplitude=waveforms_amplitude,
            peak_channels=peak_channels,
            clu_q=clu_q,
            shank_ids=shank_ids,
        )

    def concatenate(self, neurons_to_add, index_to_add=0):
        """Add two neuron spike trains together. Adds 'index_to_add' to neuron_ids, shank_ids, and peak_channels
        to help differentiate different sessions (e.g. index_to_add=100 will make the cluster_ids from
        neurons_to_add be 101, 102, 103... """
        t_start = np.min((self.t_start, neurons_to_add.t_start))
        t_stop = np.max((self.t_stop, neurons_to_add.t_stop))

        # Check to make sure everything is compatible
        assert self.sampling_rate == neurons_to_add.sampling_rate
        feature_dict = {}
        for feature in ["spiketrains", "neuron_ids", "neuron_type", "waveforms",
                        "peak_channels", "shank_ids"]:
            print(f"{feature} with kind={getattr(self, feature).dtype.kind}")
            # try:
            if feature in ["spiketrains", "neuron_type", "waveforms"]:
                feature_dict[feature] = np.concatenate((getattr(self, feature),
                                                        getattr(neurons_to_add, feature)),
                                                        axis=0)
            else:  # only add to id related fields

                feature_dict[feature] = np.concatenate((getattr(self, feature),
                                                        getattr(neurons_to_add, feature) + index_to_add),
                                                       axis=0)
            # except:
            #     print(f"Error concatenating {feature}. Set to None")
            #     feature_dict[feature] = None

        return Neurons(spiketrains=feature_dict["spiketrains"],
                       t_start=t_start,
                       t_stop=t_stop,
                       sampling_rate=self.sampling_rate,
                       neuron_ids=feature_dict["neuron_ids"],
                       neuron_type=feature_dict["neuron_type"],
                       waveforms=feature_dict["waveforms"],
                       # waveforms_amplitude=feature_dict["waveforms_amplitude"],
                       peak_channels=feature_dict["peak_channels"],
                       shank_ids=feature_dict["shank_ids"])

    def get_neuron_type(self, neuron_type):
        if isinstance(neuron_type, str):
            indices = self.neuron_type == neuron_type
        if isinstance(neuron_type, list):
            indices = np.any(
                np.vstack([ntype == self.neuron_type for ntype in neuron_type]), axis=0
            )
        return self[indices]

    def _check_integrity(self):
        assert isinstance(self.spiketrains, np.ndarray)
        # n_neurons = self.n_neurons
        # assert all(
        #     len(arr) == n_neurons
        #     for arr in [
        #         self.shankid,
        #         self.labels,
        #         self.ids,
        #         self.waveforms,
        #         self.instfiring,
        #     ]
        # )

    def __str__(self) -> str:
        return f"# neurons = {self.n_neurons}"

    def __len__(self):
        return self.n_neurons

    def add_metadata(self):
        pass

    def get_all_spikes(self):
        return np.concatenate(self.spiketrains).astype("float")

    @property
    def n_spikes(self):
        "number of spikes within each spiketrain"
        return np.asarray([len(_) for _ in self.spiketrains])

    @property
    def firing_rate(self):
        return self.n_spikes / (self.t_stop - self.t_start)

    def get_above_firing_rate(self, thresh: float):
        """Return neurons which have firing rate above thresh"""
        indices = self.firing_rate > thresh
        return self[indices]

    def get_by_id(self, ids):
        """Returns neurons object with neuron_ids equal to ids"""
        # indices = np.isin(self.neuron_ids, ids, assume_unique=True)
        indices = np.array([np.where(self.neuron_ids == _)[0][0] for _ in ids])
        return self[indices]

    def to_dataframe(self):
        """Generates a pandas dataframe with some descriptions about the neurons"""
        print("Number of neurons:", self.n_neurons)
        return pd.DataFrame(
            dict(
                neuron_type=self.neuron_type,
                neuron_id = self.neuron_ids,
                nspikes=self.n_spikes,
                mean_frate=self.firing_rate,
            )
        )

    def get_isi(self, bin_size=0.001, n_bins=200):
        """Interspike interval

        Parameters
        ----------
        bin_size : float, optional
            [description], by default 0.001
        n_bins : int, optional
            [description], by default 200

        Returns
        -------
        [type]
            [description]
        """
        bins = np.arange(n_bins + 1) * bin_size
        return np.asarray(
            [np.histogram(np.diff(spktrn), bins=bins)[0] for spktrn in self.spiketrains]
        )

    def get_waveform_similarity(self):
        waveforms = np.reshape(self.waveforms, (self.n_neurons, -1)).astype(float)
        similarity = np.corrcoef(waveforms)
        np.fill_diagonal(similarity, 0)
        return similarity

    def get_binned_spiketrains(self, bin_size=0.25, ignore_epochs: Epoch = None):
        """Get binned spike counts

        Parameters
        ----------
        bin_size : float, optional
            bin size in seconds, by default 0.25

        Returns
        -------
        neuropy.core.BinnedSpiketrains

        """
        duration = self.t_stop - self.t_start
        n_bins = np.floor(duration / bin_size)
        # bins = np.arange(self.t_start, self.t_stop + bin_size, bin_size)
        bins = np.arange(n_bins + 1) * bin_size + self.t_start
        spike_counts = np.asarray(
            [np.histogram(_, bins=bins)[0] for _ in self.spiketrains]
        ).astype("float")
        if ignore_epochs is not None:
            ignore_bins = ignore_epochs.flatten()
            ignore_indices = np.digitize(bins[:-1], ignore_bins) % 2 == 1
            spike_counts[:, ignore_indices] = np.nan

        return BinnedSpiketrain(
            spike_counts,
            t_start=self.t_start,
            bin_size=bin_size,
            neuron_ids=self.neuron_ids,
            peak_channels=self.peak_channels,
            shank_ids=self.shank_ids,
        )

    def get_mua(self, bin_size=0.001):
        """Get mua between two time points
        
        Updated to store raw spike times and corresponding neuron IDs.
        """
        # <--- NEW LOGIC START --->
        all_spikes_list = []
        all_ids_list = []

        # 1. Collect spikes and corresponding IDs
        for i, train in enumerate(self.spiketrains):
            if len(train) > 0:
                all_spikes_list.append(train)
                # Create array of same length filled with this neuron's ID
                # Assuming self.neuron_ids[i] is the ID for the ith spiketrain
                current_id = self.neuron_ids[i] 
                all_ids_list.append(np.full(len(train), current_id))

        # 2. Concatenate into single arrays
        if len(all_spikes_list) > 0:
            flat_spikes = np.concatenate(all_spikes_list)
            flat_ids = np.concatenate(all_ids_list)

            # 3. Sort by time (essential so spikes and ids line up chronologically)
            sort_idx = np.argsort(flat_spikes)
            sorted_spikes = flat_spikes[sort_idx]
            sorted_ids = flat_ids[sort_idx]
        else:
            sorted_spikes = np.array([])
            sorted_ids = np.array([])
        # <--- NEW LOGIC END --->

        # 4. Calculate binned statistics (Histogram) using the sorted spikes
        bins = np.arange(self.t_start, self.t_stop, bin_size)
        counts = stats.binned_statistic(sorted_spikes, None, bins=bins, statistic="count")[0]
        
        # 5. Return Mua with new arguments
        return Mua(
            counts.astype("int"), 
            t_start=self.t_start, 
            bin_size=bin_size,
            spikes=sorted_spikes,    # <--- Pass sorted spikes
            neuron_ids=sorted_ids    # <--- Pass corresponding IDs
        )

    def get_psth(self, t: np.array, bin_size: float, n_bins: int, n_jobs=1):
        """Get peristimulus time histograms with respect to timepoints

        Parameters
        ----------
        t : np.array
            timepoints around which psths are computed, in seconds
        bin_size : float
            binsize in seconds
        n_bins : int
            number of bins before/after the timepoints, total number of bins= 2*n_bins
        n_jobs : int, optional
            number of cpus to speed up calculations, by default 1

        Returns
        -------
        psths: shape(n_neurons, 2*n_bins, len(t))
            number of spikes for each neuron around each timepoint
        """
        n_bins_around = 2 * n_bins
        n_t = len(t)
        bins = np.linspace(-n_bins, n_bins, n_bins_around + 1) * bin_size
        t_bins = np.tile(bins, len(t)) + np.repeat(t, n_bins_around + 1)

        def get_counts(spiketimes):
            indx_right = np.searchsorted(spiketimes, t_bins[:-1], side="right")
            indx_left = np.searchsorted(spiketimes, t_bins[1:], side="left")
            # count the number of spikes and skip time bins that represent bins between adjacent time points
            counts_in_bins = np.delete(
                indx_left - indx_right,
                np.arange(n_bins_around, indx_left.size, n_bins_around + 1),
            )
            return counts_in_bins.reshape(1, n_bins_around, n_t)

        psths = Parallel(n_jobs=n_jobs)(
            delayed(get_counts)(_) for _ in self.spiketrains
        )

        return np.vstack(psths)

    def add_jitter(self):
        pass

    def get_neurons_in_epochs(self, epochs: Epoch):
        """Remove spikes that lie outside of given epochs and return a new Neurons object with t_start and t_stop changed to start of first epoch and stop of last epoch.

        Parameters
        ----------
        epochs : Epoch
            epochs defining starts and stops
        """
        assert epochs.is_overlapping == False, "epochs should be non-overlapping"
        spktrns = self.spiketrains
        epochs_bins = epochs.flatten()

        new_spktrns = []
        for spktrn in spktrns:
            bin_loc = np.digitize(spktrn, epochs_bins)
            new_spktrns.append(spktrn[bin_loc % 2 == 1])

        new_spktrns = np.array(new_spktrns, dtype="object")

        return Neurons(
            spiketrains=new_spktrns,
            t_start=epochs.starts[0],
            t_stop=epochs.stops[-1],
            sampling_rate=self.sampling_rate,
            neuron_ids=self.neuron_ids,
            neuron_type=self.neuron_type,
            waveforms=self.waveforms,
            peak_channels=self.peak_channels,
            shank_ids=self.shank_ids,
        )

    def get_modulation_in_epochs(self, epochs: Epoch, n_bins):
        """Total number of across all epochs where each epoch is divided into equal number of bins

        Parameters
        ----------
        epochs : Epoch
            epochs for calculation
        n_bins : int
            number of bins to divide each epoch

        Returns
        -------
        2d array: n_neurons x n_bins
            total number of spikes within each bin across all epochs
        """
        assert epochs.is_overlapping == False, "epochs should be non-overlapping"
        assert isinstance(n_bins, int), "n_bins can only be integer"
        starts = epochs.starts.reshape(-1, 1)
        bin_size = (epochs.durations / n_bins).reshape(-1, 1)

        # create 2D-array (n_epochs x n_bins+1) with bin_size spacing along columns
        bins = np.arange(n_bins + 1) * bin_size

        epoch_bins = (starts + bins).flatten()

        # calculate spikes on flattened epochs and delete bins which represent spike counts between (not within) epochs and then sums across all epochs for each bin
        counts = [
            np.delete(
                np.histogram(_, epoch_bins)[0],
                np.arange(n_bins, epoch_bins.size, n_bins + 1)[:-1],
            )
            .reshape(-1, n_bins)
            .sum(axis=0)
            for _ in self.spiketrains
        ]

        return np.asarray(counts)

    def get_spikes_in_epochs(
        self, epochs: Epoch, bin_size=0.01, slideby=None, sigma=None
    ):
        """
        A list of 2D arrays containing spike counts.
        
        Refactored to use np.searchsorted for high-precision, drift-free windowing.
        This removes the dependency on 1ms integer binning.

        Parameters
        ----------
        epochs : Epoch
            Start and stop times of epochs.
        bin_size : float, default 0.01
            Window size in seconds.
        slideby : float, optional
            Step size in seconds. If None, defaults to bin_size (non-overlapping).
        sigma : float, optional
            Standard deviation for gaussian kernel used for smoothing in seconds.

        Returns
        -------
        spkcount : list of np.ndarray
            List of arrays (n_neurons, n_windows).
        nbins : np.ndarray
            Number of bins within each epoch.
        """
        spkcount = []
        nbins = np.zeros(epochs.n_epochs, dtype="int")

        # If slideby is not provided, default to bin_size (non-overlapping windows)
        if slideby is None:
            slideby = bin_size

        # Iterate through each epoch
        for i, epoch in enumerate(epochs.to_dataframe().itertuples()):
            t_start = epoch.start
            t_stop = epoch.stop
            
            # 1. Generate precise floating-point time windows
            # Use a small epsilon to include the last bin if it fits exactly
            epsilon = 1e-9
            
            # Window starts: from t_start to (t_stop - bin_size)
            window_starts = np.arange(t_start, t_stop - bin_size + epsilon, slideby)
            
            # Handle cases where the epoch is shorter than one bin_size
            if len(window_starts) == 0:
                n_neurons = len(self.spiketrains)
                spkcount.append(np.zeros((n_neurons, 0)))
                nbins[i] = 0
                continue
            
            # Window ends: start + bin_size
            window_ends = window_starts + bin_size
            
            # 2. Count spikes using searchsorted (Vectorized over time windows)
            n_neurons = len(self.spiketrains)
            n_windows = len(window_starts)
            
            # Initialize matrix for this epoch
            counts_matrix = np.zeros((n_neurons, n_windows), dtype=float)
            
            for n_idx, spikes in enumerate(self.spiketrains):
                # Ensure spikes are sorted (NeuroPy data is usually sorted by time)
                # spikes = np.sort(spikes) 
                
                # Find indices where windows start
                # side='left' means we include the spike if spike_time == window_start
                idx_start = np.searchsorted(spikes, window_starts, side='left')
                
                # Find indices where windows end
                # This naturally implements the [start, end) half-open interval
                idx_end = np.searchsorted(spikes, window_ends, side='left')
                
                # Calculate counts by taking the difference of indices
                counts_matrix[n_idx] = idx_end - idx_start

            spkcount.append(counts_matrix)
            nbins[i] = n_windows

        # 3. Apply Gaussian smoothing if requested
        if sigma is not None:
            kernel = gaussian_kernel1D(sigma=sigma, bin_size=bin_size)
            spkcount = [
                np.apply_along_axis(np.convolve, arr=_, v=kernel, mode="same", axis=1)
                for _ in spkcount
            ]

        return spkcount, nbins


class BinnedSpiketrain(DataWriter):
    """Class to hold binned spiketrains"""

    def __init__(
        self,
        spike_counts: np.ndarray,
        bin_size: float,
        t_start=0.0,
        neuron_ids=None,
        peak_channels=None,
        shank_ids=None,
        metadata=None,
    ) -> None:
        super().__init__()
        self.spike_counts = spike_counts
        self.bin_size = bin_size
        self.t_start = t_start
        self.peak_channels = peak_channels
        self.shank_ids = shank_ids
        if neuron_ids is None:
            self.neuron_ids = np.arange(self.n_neurons)

        self.metadata = metadata

    @staticmethod
    def from_neurons(neurons: Neurons, t_start=None, t_stop=None, bin_size=0.25):
        pass

    @property
    def spike_counts(self):
        return self._spike_counts

    @spike_counts.setter
    def spike_counts(self, arr):
        self._spike_counts = arr

    @property
    def n_neurons(self):
        return self.spike_counts.shape[0]

    @property
    def n_bins(self):
        return self.spike_counts.shape[1]

    @property
    def duration(self):
        return self.n_bins * self.bin_size

    @property
    def t_stop(self):
        return self.t_start + self.duration

    def add_metadata(self):
        pass

    @property
    def time(self):
        return np.arange(self.n_bins) * self.bin_size + self.t_start

    def _get_nan_bins(self):
        return np.isnan(self.spike_counts).any(axis=0)

    def get_pairwise_corr(self, pairs_bool=None, return_pair_id=False):
        """Pairwise correlation between pairs of binned of spiketrains

        Parameters
        ----------
        pairs_bool : 2D bool/logical array, optional
            Only these pairs are returned, by default None which means all pairs
        return_pair_id : bool, optional
            If true pair_ids are returned, by default False

        Returns
        -------
        corr
            1d vector of pairwise correlations
        """

        assert self.n_neurons > 1, "Should have more than 1 neuron"
        corr = np.corrcoef(self.spike_counts[:, ~self._get_nan_bins()])

        if pairs_bool is not None:
            assert (
                pairs_bool.shape[0] == pairs_bool.shape[1]
            ), "pairs_bool should be sqare shpae"
            assert (
                pairs_bool.shape[0] == self.n_neurons
            ), f"pairs_bool should be of {corr.shape} shape"
            pairs_bool = pairs_bool.astype("bool")
        else:
            pairs_bool = np.ones(corr.shape).astype("bool")

        pairs_bool = np.tril(pairs_bool, k=-1)

        return corr[pairs_bool]

    @property
    def firing_rate(self):
        return self.spike_counts / self.bin_size


class Mua(DataWriter):
    def __init__(
        self,
        spike_counts: np.ndarray,
        bin_size: float,
        t_start: float = 0.0,
        spikes: np.ndarray = None,
        neuron_ids: np.ndarray = None,
        metadata=None,
    ) -> None:
        super().__init__()
        self.spike_counts = spike_counts
        self.t_start = t_start
        self.bin_size = bin_size
        self.metadata = metadata

        self.spikes = spikes if spikes is not None else np.array([])
        self.neuron_ids = neuron_ids if neuron_ids is not None else np.array([])

    @property
    def spike_counts(self):
        return self._spike_counts

    @spike_counts.setter
    def spike_counts(self, arr: np.ndarray):
        assert arr.ndim == 1, "only 1 dimensional arrays are allowed"
        self._spike_counts = arr

    @property
    def bin_size(self):
        return self._bin_size

    @bin_size.setter
    def bin_size(self, val):
        self._bin_size = val

    @property
    def n_bins(self):
        return len(self._spike_counts)

    @property
    def duration(self):
        return self.n_bins * self.bin_size

    @property
    def t_stop(self):
        return self.t_start + self.duration

    @property
    def time(self):
        return np.arange(self.n_bins) * self.bin_size + self.t_start

    @property
    def firing_rate(self):
        return self.spike_counts / self.bin_size

    def get_smoothed(self, sigma=0.02, **kwargs):
        """Smoothing of mua spike counts

        Parameters
        ----------
        sigma : float, optional
            gaussian kernel in seconds, by default 0.02 s (20 milliseconds)
        kwargs : float, optional
            keyword arguments for scipy.ndimage.gaussian_filter1d, by default 4.0

        Returns
        -------
        core.MUA object
            containing smoothed spike counts
        """

        dt = self.bin_size
        spike_counts = gaussian_filter1d(
            self.spike_counts, sigma=sigma / dt, output="float", **kwargs
        )
        return Mua(spike_counts, t_start=self.t_start, bin_size=self.bin_size)

    def time_slice(self, t_start, t_stop):
        # Slice the binned counts (unchanged)
        indices = (self.time >= t_start) & (self.time <= t_stop)
        
        # Slice the raw spikes and ids (NEW)
        raw_mask = (self.spikes >= t_start) & (self.spikes <= t_stop)

        return Mua(
            spike_counts=self.spike_counts[indices],
            bin_size=self.bin_size,
            t_start=t_start,
            spikes=self.spikes[raw_mask],          # <--- Pass sliced spikes
            neuron_ids=self.neuron_ids[raw_mask]   # <--- Pass sliced IDs
        )

    def to_dataframe(self):
        return pd.DataFrame({"time": self.time, "spike_counts": self.spike_counts})


def pe_raster(
    neurons: Neurons,
    neuron_id: int,
    event_times: np.ndarray or list,
    buffer_sec=(5, 5),
):
    """Get peri-event raster of spike times"""
    spiketrain = neurons.spiketrains[neuron_id]
    rast = []
    for event_time in event_times:
        time_bool = (spiketrain > (event_time - buffer_sec[0])) & (
            spiketrain <= (event_time + buffer_sec[1])
        )
        rast.append(spiketrain[time_bool] - event_time)

    return Neurons(rast, t_stop=buffer_sec[1], t_start=-buffer_sec[0])


def binned_pe_raster(
    binned_spiketrain: (BinnedSpiketrain, Mua),
    event_times: np.ndarray or list,
    neuron_id: int = 0,
    buffer_sec=(5, 5),
):
    """Build a peri-event raster for a binned spiketrain or MUA. neuron_id only needed for
    binned_spiketrain class."""

    if isinstance(binned_spiketrain, BinnedSpiketrain):
        binned_fr = binned_spiketrain.firing_rate[neuron_id]
    elif isinstance(binned_spiketrain, Mua):
        binned_fr = binned_spiketrain.firing_rate

    firing_rate = []
    for event_time in event_times:
        time_bool = (binned_spiketrain.time > (event_time - buffer_sec[0])) & (
            binned_spiketrain.time
            <= (event_time + buffer_sec[1] + binned_spiketrain.bin_size * 0.5)
        )
        firing_rate.append(binned_fr[time_bool])

    fr_len = [len(f) for f in firing_rate]
    if np.max(fr_len) == np.min(fr_len):
        fr_array = np.array(firing_rate)
    elif (
        np.max(fr_len) - np.min(fr_len)
    ) == 1:  # append a 0 firing rate to last bin of any short
        for id in np.where(fr_len == np.min(fr_len))[0]:
            firing_rate[id] = np.append(firing_rate[id], np.nan)
        fr_array = np.array(firing_rate)
    else:
        fr_array = np.nan
        print(
            "Raster has uneven # of bins in one row, likely due to edge effects. Fix code or delete start/end event from input"
        )

    pe_times = np.arange(
        -buffer_sec[0],
        buffer_sec[1] + binned_spiketrain.bin_size * 0.5,
        binned_spiketrain.bin_size,
    )

    return fr_array, pe_times
