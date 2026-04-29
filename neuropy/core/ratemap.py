import numpy as np
from . import DataWriter
from scipy import stats, interpolate
from scipy.ndimage import gaussian_filter1d


class Ratemap(DataWriter):
    """
    Ratemap class for storing and manipulating 1D neural tuning curves.

    This class represents spatial firing rate maps of neurons for linear track
    experiments. Commonly used in neuroscience to analyze place cells and other
    spatially-tuned neurons.

    Properties:
        tuning_curves (np.ndarray): Firing rates for each neuron across spatial bins, shape (n_neurons, n_bins)
        coords (np.ndarray): Spatial coordinates in cm, shape (n_bins,)
        occupancy (np.ndarray): Time spent in each spatial bin, shape (n_bins,)
        neuron_ids (np.ndarray): Unique identifiers for each neuron, shape (n_neurons,)
        metadata (dict): Additional information about the recording session
        n_neurons (int): Number of neurons in the ratemap [read-only]
        n_bins (int): Number of spatial bins [read-only]
        x_binsize (float): Physical size of bins (cm) [read-only]

    Methods:
        copy(): Creates a deep copy of the Ratemap object
        neuron_slice(inds=None, ids=None): Extracts subset of neurons by index or ID
        resample(nbins): Resamples ratemap to different number of bins
        smooth_tuning_curves(sigma_bin, mode): Applies Gaussian smoothing to tuning curves
        peak_locations(by='index'): Finds spatial location of peak firing for each neuron
        get_sort_order(by='index'): Returns neuron ordering sorted by peak firing location
        peak_firing_rate(): Returns maximum firing rate for each neuron
        get_frate_normalized(method): Returns normalized firing rates
    """

    def __init__(
        self,
        tuning_curves: np.ndarray,
        coords: np.ndarray,
        occupancy=None,
        neuron_ids=None,
        metadata=None,
    ) -> None:
        """
        Initialize a Ratemap object.

        Parameters
        ----------
        tuning_curves : np.ndarray, shape (n_neurons, n_bins)
            Firing rates for each neuron across spatial bins.
        coords : float or array-like
            Spatial coordinates in cm:
                * float: uniform bin spacing
                * array: explicit bin center coordinates
        occupancy : np.ndarray, optional
            Time spent in each spatial bin, shape (n_bins,)
        neuron_ids : np.ndarray, optional
            Unique identifiers for each neuron
        metadata : dict, optional
            Additional metadata for the ratemap
        """
        super().__init__(metadata=metadata)
        self.tuning_curves = tuning_curves
        self.coords = coords
        self.occupancy = occupancy
        self.neuron_ids = neuron_ids

    @property
    def tuning_curves(self):
        """Get the tuning curves array."""
        return self._tuning_curves

    @tuning_curves.setter
    def tuning_curves(self, val):
        val = np.asarray(val)
        if val.ndim == 1:
            val = val.reshape(1, -1)
        # CHANGE: Allow 2 or 3 dimensions
        assert val.ndim in (2, 3), "tuning_curves must be 2D (N, Bins) or 3D (N, Bins, Dirs)"
        self._tuning_curves = val

    @property
    def occupancy(self):
        """Get the occupancy map."""
        return self._occupancy

    @occupancy.setter
    def occupancy(self, arr):
        """
        Set and validate occupancy map.
        
        Occupancy must match the number of spatial bins.
        """
        if arr is not None:
            arr = np.asarray(arr)
            assert arr.shape == (self.n_bins,), \
                f"Occupancy shape {arr.shape} must match n_bins ({self.n_bins},)"
        self._occupancy = arr

    @property
    def coords(self):
        """Get the coordinate array."""
        return self._coords

    @coords.setter
    def coords(self, val):
        """
        Set and validate spatial coordinates.
        
        Handles multiple input formats and ensures coordinates are equally spaced.
        """
        if isinstance(val, (int, float)):
            # Convert bin size to coordinate array
            val = np.arange(0, val * self.n_bins, val)
        
        val = np.asarray(val).flatten()
        
        assert len(val) == self.n_bins, \
            f"Coords length {len(val)} must equal n_bins {self.n_bins}"
        
        # Verify equal spacing
        diffs = np.diff(val)
        assert np.allclose(diffs, diffs[0]), "Coordinates must be equally spaced"
        
        self._coords = val

    def x_coords(self):
        """Alias for coords (for API compatibility)."""
        return self._coords

    @property
    def neuron_ids(self):
        """Get neuron IDs."""
        return self._neuron_ids

    @neuron_ids.setter
    def neuron_ids(self, arr):
        """
        Set and validate neuron IDs.
        
        If not provided, defaults to sequential integers [0, 1, 2, ...].
        """
        if arr is not None:
            assert len(arr) == self.n_neurons, \
                f"neuron_ids length {len(arr)} must match n_neurons {self.n_neurons}"
            self._neuron_ids = np.asarray(arr)
        else:
            self._neuron_ids = np.arange(self.n_neurons)

    @property
    def x_binsize(self):
        """Get the bin size (cm)."""
        return np.diff(self.coords)[0]

    @property
    def n_neurons(self):
        """Get the number of neurons in the ratemap."""
        return self._tuning_curves.shape[0]

    @property
    def n_bins(self):
        """Get the number of spatial bins."""
        return self._tuning_curves.shape[1]

    def copy(self):
        """
        Create a deep copy of the Ratemap object.
        
        Returns
        -------
        Ratemap
            Independent copy with the same data
        """
        return Ratemap(
            tuning_curves=self.tuning_curves.copy(),
            coords=self.coords.copy(),
            neuron_ids=self.neuron_ids.copy(),
            occupancy=self.occupancy.copy() if self.occupancy is not None else None,
            metadata=self.metadata.copy() if self.metadata is not None else None,
        )

    def neuron_slice(self, inds=None, ids=None):
        """
        Extract a subset of neurons by index or ID.
        
        Parameters
        ----------
        inds : array-like, optional
            Integer indices of neurons to select (0-based)
        ids : array-like, optional
            Neuron IDs to select (must match values in self.neuron_ids)
            
        Returns
        -------
        Ratemap
            New Ratemap containing only selected neurons
            
        Notes
        -----
        Exactly one of 'inds' or 'ids' must be specified, not both.
        """
        assert (inds is None) != (ids is None), \
            "Specify exactly one of 'inds' (indices) or 'ids' (neuron IDs)"
        
        if ids is not None:
            ids = np.asarray(ids)
            inds = np.array([np.where(idd == self.neuron_ids)[0][0] for idd in ids])
        
        inds = np.sort(inds)
        
        return Ratemap(
            tuning_curves=self.tuning_curves[inds],
            coords=self.coords.copy(),
            neuron_ids=self.neuron_ids[inds],
            occupancy=self.occupancy,
            metadata=self.metadata,
        )

    def get_frate_normalized(self, method='zscore'):
        """
        Get normalized firing rates.
        
        Parameters
        ----------
        method : str, default 'zscore'
            Normalization method:
            - 'zscore': Z-score normalization across spatial bins
            - 'peak': Divide by peak firing rate (scales to [0, 1])
            - 'mean': Divide by mean firing rate
            - 'minmax': Min-max normalization to [0, 1]
            
        Returns
        -------
        np.ndarray
            Normalized tuning curves, shape (n_neurons, n_bins)
        """
        if method == 'zscore':
            return stats.zscore(self.tuning_curves, axis=1)
        
        elif method == 'peak':
            peak_rates = self.peak_firing_rate()
            peak_rates = np.where(peak_rates == 0, 1, peak_rates)
            return self.tuning_curves / peak_rates[:, np.newaxis]
        
        elif method == 'mean':
            mean_rates = np.mean(self.tuning_curves, axis=1, keepdims=True)
            mean_rates = np.where(mean_rates == 0, 1, mean_rates)
            return self.tuning_curves / mean_rates
        
        elif method == 'minmax':
            min_vals = np.min(self.tuning_curves, axis=1, keepdims=True)
            max_vals = np.max(self.tuning_curves, axis=1, keepdims=True)
            range_vals = max_vals - min_vals
            range_vals = np.where(range_vals == 0, 1, range_vals)
            return (self.tuning_curves - min_vals) / range_vals
        
        else:
            raise ValueError(f"Unknown method: {method}. "
                            f"Choose from: 'zscore', 'peak', 'mean', 'minmax'")

    def resample(self, nbins):
        """
        Resample ratemap to a different number of bins using interpolation.
        
        Parameters
        ----------
        nbins : int
            Target number of spatial bins
            
        Returns
        -------
        Ratemap
            New Ratemap with resampled tuning curves
        """
        f_tc = interpolate.interp1d(self.coords, self.tuning_curves, axis=1)
        x_new = np.linspace(self.coords[0], self.coords[-1], nbins)
        tc_new = f_tc(x_new)

        return Ratemap(
            tuning_curves=tc_new,
            coords=x_new,
            neuron_ids=self.neuron_ids.copy(),
            occupancy=None,  # Occupancy not interpolated
            metadata=self.metadata.copy() if self.metadata else None,
        )

    @staticmethod
    def _circular_gaussian_smooth(tuning, sigma, truncate=4.0):
        """
        Smooth 1D circular tuning curves with Gaussian kernel using SciPy's built-in wrap mode.
        
        Parameters
        ----------
        tuning : np.ndarray, shape (n_neurons, n_bins)
        sigma : float
            Gaussian kernel standard deviation in bins
        truncate : float, default 4.0
            Truncate filter at this many sigmas (matches scipy default)
        """
        from scipy.ndimage import gaussian_filter1d
        
        # mode='wrap' exactly implements circular boundary conditions
        return gaussian_filter1d(tuning, sigma=sigma, axis=1, mode='wrap', truncate=truncate)
    
    def smooth_tuning_curves(self, sigma_bin, mode='linear'):
        """
        Apply Gaussian smoothing to tuning curves.
        
        Parameters
        ----------
        sigma_bin : float
            Standard deviation of Gaussian kernel in bin units.
        mode : str, default 'linear'
            'linear': Standard Gaussian smoothing (edge padding)
            'circular': Circular smoothing with edge wrapping
            
        Returns
        -------
        np.ndarray
            Smoothed tuning curves, shape (n_neurons, n_bins)
        """
        if sigma_bin <= 0:
            return self.tuning_curves.copy()
        
        sigma = sigma_bin / self.x_binsize
        
        if mode == 'linear':
            return gaussian_filter1d(self.tuning_curves, sigma=sigma, axis=1)
        elif mode == 'circular':
            return self._circular_gaussian_smooth(self.tuning_curves, sigma)
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'linear' or 'circular'")

    def peak_locations(self, by="index", mode="linear", n_interp=1000):
        """
        Find the spatial location of peak firing.
        
        Parameters
        ----------
        by : str, default "index"
            - "index": Returns integer bin index of max firing (NO interpolation).
            - "position": Returns physical position using high-res interpolation.
        mode : str, default "linear"
            - "linear": Standard PCHIP interpolation (used if by="position").
            - "circular": Wraps data 0..2pi (used if by="position").
        n_interp : int, default 1000
            Number of points to interpolate (used if by="position").
            
        Returns
        -------
        np.ndarray
            Peak location for each neuron.
        """
        n_neurons, n_bins = self.tuning_curves.shape
        
        # === Case 1: Simple Index (Original Method, No Interpolation) ===
        if by == "index":
            return np.argmax(self.tuning_curves, axis=1)

        # === Case 2: Position (With Interpolation) ===
        elif by == "position":
            if mode == "circular":
                # 1. Close the loop: Append the first column to the end
                y_data = np.column_stack([self.tuning_curves, self.tuning_curves[:, 0]])
                
                # 2. Define x-axis (0 to 2pi inclusive)
                x_data = np.linspace(0, 2 * np.pi, n_bins + 1, endpoint=True)
                
                # 3. Fit PCHIP & Create High-Res Grid
                interpolator = interpolate.PchipInterpolator(x_data, y_data, axis=1)
                x_high_res = np.linspace(0, 2 * np.pi, n_interp, endpoint=False)
                
                # 4. Find Peaks
                y_high_res = interpolator(x_high_res)
                peak_indices_high_res = np.argmax(y_high_res, axis=1)
                peak_locs = x_high_res[peak_indices_high_res]
                
                # Return peak locations in radians (0..2pi)
                return peak_locs
                
            elif mode == "linear":
                # Standard PCHIP using physical coordinates
                y_data = self.tuning_curves
                x_data = self.coords
                
                interpolator = interpolate.PchipInterpolator(x_data, y_data, axis=1)
                x_high_res = np.linspace(x_data[0], x_data[-1], n_interp)
                
                y_high_res = interpolator(x_high_res)
                peak_indices_high_res = np.argmax(y_high_res, axis=1)
                peak_locs = x_high_res[peak_indices_high_res]
                
                return peak_locs
            else:
                raise ValueError(f"Invalid mode: {mode}. Use 'linear' or 'circular'.")
        
        else:
            raise ValueError(f"Invalid 'by': {by}. Use 'index' or 'position'")

    def get_sort_order(self, by="index", mode="linear"):
        """
        Get neuron ordering sorted by peak firing location.
        """
        peak_locs = self.peak_locations(by="index", mode=mode)
        sort_ind = np.argsort(peak_locs)
        
        if by == "neuron_id":
            return self.neuron_ids[sort_ind]
        elif by == "index":
            return sort_ind
        else:
            raise ValueError(f"Invalid 'by': {by}. Use 'index' or 'neuron_id'")