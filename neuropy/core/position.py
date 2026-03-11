import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from typing import Optional, Union, Dict, Any
from ..utils import mathutil
from .epoch import Epoch
from .datawriter import DataWriter
from neuropy.utils.detect import detect_epochs_peak


class Position(DataWriter):
    """
    Represents position data with optional rotation in 1D, 2D, or 3D space.
    
    This class handles spatial tracking data commonly used in neuroscience,
    such as animal position in behavioral arenas or head direction during
    navigation tasks.
    
    Attributes:
        traces: Position coordinates as numpy array, shape (n_dims, n_samples)
        traces_rot: Optional rotation data, shape (n_dims, n_samples)
        sampling_rate: Sampling frequency in Hz
        t_start: Start time of recording in seconds
        
    Examples:
        # 1D position (linear track)
        pos = Position(traces=np.array([0, 1, 2, 3, 4]), sampling_rate=30)
        
        # 2D position (open field)
        traces_2d = np.array([[0, 1, 2], [0, 1, 2]])  # x and y coordinates
        pos = Position(traces=traces_2d, sampling_rate=30)
        
        # Create from DataFrame
        df = pd.DataFrame({'x': [0, 1, 2], 'y': [0, 1, 2]})
        pos = Position.from_dataframe(df, sampling_rate=30)
    """
    
    def __init__(
        self,
        traces: np.ndarray,
        traces_rot: Optional[np.ndarray] = None,
        t_start: float = 0,
        sampling_rate: float = 120,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize Position object.
        
        Args:
            traces: Position coordinates array. Can be:
                   - 1D array: shape (n_samples,) for 1D position
                   - 2D array: shape (n_dims, n_samples) for multi-dimensional
                   Maximum of 3 spatial dimensions supported.
            traces_rot: Rotation data (optional), same shape constraints as traces
            t_start: Start time in seconds (default: 0)
            sampling_rate: Sampling frequency in Hz (default: 120)
            metadata: Additional metadata dictionary (optional)
            
        Raises:
            AssertionError: If traces exceed 3 dimensions
            
        Examples:
            # 1D position
            pos = Position(np.array([0, 1, 2, 3]), sampling_rate=30)
            
            # 2D position with rotation
            pos = Position(
                traces=np.array([[0, 1, 2], [0, 1, 2]]),
                traces_rot=np.array([[0, 0.1, 0.2]]),  # heading angle
                sampling_rate=30
            )
        """
        # Ensure traces is 2D array for consistent internal representation
        # 1D input [1,2,3] becomes [[1,2,3]] (shape: 1 x n_samples)
        traces = self._ensure_2d_array(traces)
        self._validate_dimensions(traces, max_dims=3, name="position")
        
        # Store position data
        self.traces = traces
        self._t_start = t_start
        self._sampling_rate = sampling_rate
        
        # Handle rotation data if provided
        self.traces_rot = self._process_rotation_traces(traces_rot)
        
        # Initialize parent class (DataWriter) for save/load functionality
        super().__init__(metadata=metadata)
    
    def __repr__(self) -> str:
        """
        String representation of Position object.
        
        Provides a comprehensive summary of the Position object including
        dimensionality, number of frames, time range, duration, sampling rate,
        and whether rotation data is present.
        
        Returns:
            Formatted string with Position object information
        
        Examples:
            >>> pos = Position(traces=np.random.rand(2, 1000), sampling_rate=30)
            >>> print(pos)
            Position(ndim=2, n_frames=1000, t=[0.00-33.33]s, duration=33.33s, rate=30.0Hz, has_rotation=False)
        """
        has_rotation = self.traces_rot is not None
        return (
            f"Position(ndim={self.ndim}, n_frames={self.n_frames}, "
            f"t=[{self.t_start:.2f}-{self.t_stop:.2f}]s, "
            f"duration={self.duration:.2f}s, rate={self.sampling_rate:.1f}Hz, "
            f"has_rotation={has_rotation})"
        )
    
    # ==================== Static Helper Methods ====================
    
    @staticmethod
    def _ensure_2d_array(arr: np.ndarray) -> np.ndarray:
        """
        Convert 1D array to 2D with shape (1, n_samples).
        
        This ensures consistent internal representation regardless of input format.
        
        Args:
            arr: Input array, can be 1D or 2D
            
        Returns:
            2D array with shape (n_dims, n_samples)
            
        Examples:
            [1, 2, 3] -> [[1, 2, 3]]  (shape: 1 x 3)
            [[1, 2], [3, 4]] -> [[1, 2], [3, 4]]  (unchanged)
        """
        return arr.reshape(1, -1) if arr.ndim == 1 else arr
    
    @staticmethod
    def _validate_dimensions(arr: np.ndarray, max_dims: int, name: str) -> None:
        """
        Validate that array doesn't exceed maximum spatial dimensions.
        
        Args:
            arr: Array to validate, shape (n_dims, n_samples)
            max_dims: Maximum allowed dimensions
            name: Name for error message (e.g., "position" or "rotation")
            
        Raises:
            AssertionError: If array exceeds max_dims
        """
        if arr.shape[0] > max_dims:
            raise AssertionError(
                f"Maximum dimension of {name} is {max_dims}, got {arr.shape[0]}"
            )
    
    def _process_rotation_traces(self, traces_rot: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Process and validate rotation traces.
        
        Args:
            traces_rot: Rotation data or None
            
        Returns:
            Processed 2D rotation array or None
        """
        if not isinstance(traces_rot, np.ndarray):
            return None
        
        traces_rot = self._ensure_2d_array(traces_rot)
        self._validate_dimensions(traces_rot, max_dims=3, name="rotation")
        return traces_rot
    
    # ==================== Position Coordinate Properties ====================
    
    @property
    def x(self) -> np.ndarray:
        """
        Get x-coordinate (first dimension).
        
        Returns:
            1D array of x positions
        """
        return self.traces[0]
    
    @x.setter
    def x(self, value: np.ndarray) -> None:
        """Set x-coordinate."""
        self.traces[0] = value
    
    @property
    def y(self) -> np.ndarray:
        """
        Get y-coordinate (second dimension).
        
        Returns:
            1D array of y positions
            
        Raises:
            AssertionError: If position is 1D (no y-coordinate)
        """
        if self.ndim <= 1:
            raise AssertionError("No y-coordinate for one-dimensional position")
        return self.traces[1]
    
    @y.setter
    def y(self, value: np.ndarray) -> None:
        """Set y-coordinate."""
        if self.ndim <= 1:
            raise AssertionError("Position data has only one dimension")
        self.traces[1] = value
    
    @property
    def z(self) -> np.ndarray:
        """
        Get z-coordinate (third dimension).
        
        Returns:
            1D array of z positions
            
        Raises:
            AssertionError: If position is not 3D
        """
        if self.ndim != 3:
            raise AssertionError("Position data is not three-dimensional")
        return self.traces[2]
    
    @z.setter
    def z(self, value: np.ndarray) -> None:
        """Set z-coordinate."""
        if self.ndim != 3:
            raise AssertionError("Position data is not three-dimensional")
        self.traces[2] = value
    
    # ==================== Rotation Coordinate Properties ====================
    
    @property
    def x_rot(self) -> np.ndarray:
        """Get x-rotation (roll)."""
        if self.traces_rot is None:
            raise AssertionError("No rotation data available")
        return self.traces_rot[0]
    
    @x_rot.setter
    def x_rot(self, value: np.ndarray) -> None:
        """Set x-rotation."""
        if self.traces_rot is None:
            raise AssertionError("No rotation data available")
        self.traces_rot[0] = value
    
    @property
    def y_rot(self) -> np.ndarray:
        """Get y-rotation (pitch)."""
        if self.traces_rot is None:
            raise AssertionError("No rotation data available")
        if self.traces_rot.shape[0] <= 1:
            raise AssertionError("No y-rotation for one-dimensional rotation")
        return self.traces_rot[1]
    
    @y_rot.setter
    def y_rot(self, value: np.ndarray) -> None:
        """Set y-rotation."""
        if self.traces_rot is None:
            raise AssertionError("No rotation data available")
        if self.traces_rot.shape[0] <= 1:
            raise AssertionError("Rotation data has only one dimension")
        self.traces_rot[1] = value
    
    @property
    def z_rot(self) -> np.ndarray:
        """Get z-rotation (yaw/heading direction)."""
        if self.traces_rot is None:
            raise AssertionError("No rotation data available")
        if self.traces_rot.shape[0] != 3:
            raise AssertionError("Rotation data is not three-dimensional")
        return self.traces_rot[2]
    
    @z_rot.setter
    def z_rot(self, value: np.ndarray) -> None:
        """Set z-rotation."""
        if self.traces_rot is None:
            raise AssertionError("No rotation data available")
        if self.traces_rot.shape[0] != 3:
            raise AssertionError("Rotation data is not three-dimensional")
        self.traces_rot[2] = value
    
    # ==================== Time-Related Properties ====================
    
    @property
    def t_start(self) -> float:
        """Get start time in seconds."""
        return self._t_start
    
    @t_start.setter
    def t_start(self, value: float) -> None:
        """Set start time in seconds."""
        self._t_start = value
    
    @property
    def n_frames(self) -> int:
        """
        Get number of time samples/frames.
        
        Returns:
            Number of temporal samples in the recording
        """
        return self.traces.shape[1]
    
    @property
    def duration(self) -> float:
        """
        Get total duration of recording in seconds.
        
        Returns:
            Duration = n_frames / sampling_rate
        """
        return self.n_frames / self.sampling_rate
    
    @property
    def t_stop(self) -> float:
        """
        Get end time in seconds.
        
        Returns:
            Last timestamp in the time array
        """
        return self.time[-1]
    
    @property
    def time(self) -> np.ndarray:
        """
        Get time array for all samples.
        
        Auto-generated from sampling rate and start time.
        
        Returns:
            1D array of timestamps in seconds
        """
        time_step = 1 / self.sampling_rate
        return np.arange(self.n_frames) * time_step + self.t_start
    
    # ==================== Shape and Sampling Properties ====================
    
    @property
    def ndim(self) -> int:
        """
        Get number of spatial dimensions.
        
        Returns:
            1, 2, or 3 depending on position data shape
        """
        return self.traces.shape[0]
    
    @property
    def sampling_rate(self) -> float:
        """Get sampling rate in Hz."""
        return self._sampling_rate
    
    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:
        """Set sampling rate in Hz."""
        self._sampling_rate = value
    
    # ==================== Motion Properties ====================
    
    @property
    def speed(self) -> np.ndarray:
        """
        Calculate instantaneous speed (magnitude of velocity).
        Speed is always non-negative and represents the rate of position change
        regardless of direction. Computed as the Euclidean distance traveled
        per time step.
        
        Returns:
            1D array of speeds in position_units/second.
            First value is 0 (no previous position to compare).
        
        Examples:
            For 2D position moving from (0,0) to (3,4):
            distance = sqrt(3² + 4²) = 5 units
            speed = 5 / time_step
        """
        time_step = 1 / self.sampling_rate
        
        # Calculate position differences between consecutive samples
        # For 2D: diff([[x1,x2,x3], [y1,y2,y3]]) = [[x2-x1,x3-x2], [y2-y1,y3-y2]]
        displacement = np.diff(self.traces, axis=1)
        
        # Calculate Euclidean distance (displacement magnitude)
        # sqrt(dx² + dy² + dz²) for each time step
        distance = np.sqrt((displacement ** 2).sum(axis=0))
        
        # Convert displacement to velocity (distance per second)
        speed = distance / time_step
        
        # Prepend 0 for first sample (no previous position)
        return np.hstack(([0], speed))

    @property
    def velocity(self) -> np.ndarray:
        """
        Calculate directional velocity for each dimension.
        
        Unlike speed (which is always positive), velocity includes direction:
        - 1D: Positive = moving forward, negative = moving backward
        - 2D: Returns (vx, vy) - velocity in x and y directions
        - 3D: Returns (vx, vy, vz) - velocity in x, y, and z directions
        
        Returns:
            For 1D: 1D array of signed velocities
            For 2D/3D: 2D array, shape (n_dims, n_samples)
                    First column is zeros (no previous position)
        
        Examples:
            # 1D position
            pos = Position([0, 1, 3, 2], sampling_rate=1)
            pos.velocity  # [0, 1, 2, -1] (positive = forward, negative = back)
            
            # 2D position  
            pos = Position([[0, 1, 2], [0, 2, 1]], sampling_rate=1)
            pos.velocity  # [[0, 1, 1],    # vx
                        #  [0, 2, -1]]   # vy
        """
        time_step = 1 / self.sampling_rate
        
        # Calculate change in position for each dimension
        # Shape: (n_dims, n_samples-1)
        position_change = np.diff(self.traces, axis=1)
        
        # Convert to velocity (change per second)
        velocity = position_change / time_step
        
        # Prepend zeros for first sample
        zeros_column = np.zeros((self.ndim, 1))
        velocity_with_initial = np.hstack((zeros_column, velocity))
        
        # For 1D, return as 1D array for backward compatibility
        if self.ndim == 1:
            return velocity_with_initial[0]
        
        return velocity_with_initial
    
    # ==================== Data Processing Methods ====================
    
    def get_smoothed(self, sigma_t: float) -> 'Position':
        """
        Apply Gaussian smoothing to position data.
        
        Useful for removing noise from tracking data while preserving
        overall trajectory shape.
        
        Args:
            sigma_t: Standard deviation for Gaussian kernel in seconds.
                      Larger values = more smoothing.
                      Typical values: 0.1 - 1.0 seconds

        Returns:
            New Position object with smoothed traces
            
        Examples:
            # Smooth position with 0.5 second window
            smoothed_pos = pos.get_smoothed(sigma=0.5)
        """
        time_step = 1.0 / self.sampling_rate
        sigma_samples = sigma_t / time_step  # Convert seconds to samples
        
        def apply_smoothing(data):
            """Apply Gaussian filter along time axis."""
            return gaussian_filter1d(data, sigma=sigma_samples, axis=-1)
        
        # Smooth position traces
        smoothed_traces = apply_smoothing(self.traces)
        
        # Smooth rotation traces if present
        smoothed_rotation = None
        if self.traces_rot is not None:
            smoothed_rotation = apply_smoothing(self.traces_rot)
        
        return Position(
            traces=smoothed_traces,
            traces_rot=smoothed_rotation,
            sampling_rate=self.sampling_rate,
            t_start=self.t_start,
        )
    
    # ==================== DataFrame Conversion ====================
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert position data to pandas DataFrame.
        
        The DataFrame will contain:
        - 'time' column (always included)
        - Coordinate columns based on dimensionality:
          * 1D: 'x' only
          * 2D: 'x', 'y'
          * 3D: 'x', 'y', 'z'
        - 'speed' column (always included)
        - Velocity columns ('vx', 'vy', 'vz') for each dimension
        
        Returns:
            DataFrame with time, coordinates, and motion data.
            Metadata stored in df.attrs['metadata']
            
        Examples:
            df = pos.to_dataframe()
            print(df.columns)  # ['time', 'x', 'y', 'speed', 'vx', 'vy']
            print(df.attrs['metadata'])  # {'t_start': 0, 't_stop': 10, ...}
        """
        data_dict = {"time": self.time}
        
        # Add coordinate columns based on available dimensions
        coord_names = ["x", "y", "z"]
        for i in range(self.ndim):
            coord_name = coord_names[i]
            data_dict[coord_name] = self.traces[i]
        
        # Add speed (always available)
        data_dict["speed"] = self.speed
        
        # Add velocity components for each dimension
        velocity_names = ["vx", "vy", "vz"]
        velocity_data = self.velocity
        
        # Handle 1D case (velocity is 1D array)
        if self.ndim == 1:
            data_dict["vx"] = velocity_data
        else:
            # 2D/3D case (velocity is 2D array)
            for i in range(self.ndim):
                data_dict[velocity_names[i]] = velocity_data[i]
        
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        
        # Attach metadata as DataFrame attributes
        metadata_attrs = {
            "t_start": self.t_start,
            "t_stop": self.t_stop,
            "sampling_rate": self.sampling_rate,
            "ndim": self.ndim,
        }
        df.attrs["metadata"] = metadata_attrs
        
        return df
    
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        sampling_rate: float = 120,
        t_start: float = 0
    ) -> 'Position':
        """
        Create Position object from pandas DataFrame.
        
        Alternative constructor that accepts DataFrames with coordinate columns.
        Automatically detects dimensionality based on available columns.
        
        Args:
            df: DataFrame containing coordinate columns:
                - Required: at least 'x' column
                - Optional: 'y' column (for 2D)
                - Optional: 'z' column (for 3D)
            sampling_rate: Sampling frequency in Hz (default: 120)
                          Overridden by df.attrs['metadata']['sampling_rate'] if present
            t_start: Start time in seconds (default: 0)
                    Overridden by df.attrs['metadata']['t_start'] if present
            
        Returns:
            New Position object
            
        Raises:
            ValueError: If DataFrame lacks required 'x' column
            
        Examples:
            # 1D from DataFrame
            df = pd.DataFrame({'x': [0, 1, 2, 3]})
            pos = Position.from_dataframe(df, sampling_rate=30)
            
            # 2D from DataFrame
            df = pd.DataFrame({'x': [0, 1, 2], 'y': [0, 1, 2]})
            pos = Position.from_dataframe(df, sampling_rate=30)
            
            # 3D from DataFrame
            df = pd.DataFrame({'x': [0, 1], 'y': [0, 1], 'z': [0, 1]})
            pos = Position.from_dataframe(df, sampling_rate=30)
        """
        # Determine available dimensions from DataFrame columns
        coordinate_columns = ['x', 'y', 'z']
        available_coords = [col for col in coordinate_columns if col in df.columns]
        
        if not available_coords:
            raise ValueError("DataFrame must contain at least an 'x' column")
        
        # Build traces array based on dimensionality
        if len(available_coords) == 1:
            # 1D: Keep as 1D array, will be converted to 2D in __init__
            traces = df[available_coords[0]].to_numpy()
        else:
            # 2D or 3D: Stack coordinate arrays vertically
            coord_arrays = [df[col].to_numpy() for col in available_coords]
            traces = np.vstack(coord_arrays)
        
        # Extract metadata from DataFrame attributes if present
        metadata = df.attrs.get('metadata', {})
        final_sampling_rate = metadata.get('sampling_rate', sampling_rate)
        final_t_start = metadata.get('t_start', t_start)
        
        # Debug output
        n_samples = traces.shape[1] if traces.ndim > 1 else traces.shape[0]
        print(f"Creating Position from DataFrame:")
        print(f"  Dimensions: {len(available_coords)}D ({', '.join(available_coords)})")
        print(f"  Samples: {n_samples}")
        print(f"  Sampling rate: {final_sampling_rate} Hz")
        print(f"  Start time: {final_t_start} s")
        
        return cls(
            traces,
            t_start=final_t_start,
            sampling_rate=final_sampling_rate,
            metadata=metadata
        )
    
    # ==================== Temporal Slicing ====================
    
    def time_slice(
        self,
        t_start: float,
        t_stop: float,
        zero_times: bool = False
    ) -> 'Position':
        """
        Extract a temporal slice of position data.
        
        Creates a new Position object containing only data within the
        specified time window. Useful for analyzing specific behavioral epochs.
        
        Args:
            t_start: Start time for slice in seconds
            t_stop: Stop time for slice in seconds
            zero_times: If True, reset time to start at 0 in the sliced data
                       If False, preserve original timestamps
            
        Returns:
            New Position object containing sliced data
            
        Examples:
            # Extract data from 10-20 seconds
            sliced = pos.time_slice(10, 20)
            
            # Extract and reset time to start at 0
            sliced = pos.time_slice(10, 20, zero_times=True)
        """
        # Get boolean index for time window (from parent DataWriter)
        slice_indices = super()._time_slice_params(t_start, t_stop)
        
        # Determine new start time
        new_t_start = 0 if zero_times else t_start
        
        return Position(
            traces=self.traces[:, slice_indices],
            traces_rot=self.traces_rot[:, slice_indices] if self.traces_rot is not None else None,
            t_start=new_t_start,
            sampling_rate=self.sampling_rate,
        )
    

class CircularPosition(Position):
    """
    Extended Position class for circular track experiments.
    
    Requires 1D position data representing angular position on a circular track.
    
    Inherits from Position:
    ----------------------
    x : angular coordinate
    t_start, t_stop, time : temporal information
    n_frames, duration : data dimensions
    sampling_rate : temporal sampling rate
    to_dataframe, from_dataframe : data conversion methods
    
    Overridden/Extended:
    -------------------
    radius : track radius property (cm) 91.44 cm = 3 ft for 6 ft diameter
    velocity : linearized velocity accounting for circular wrapping
    speed : linearized speed accounting for circular wrapping
    time_slice : returns CircularPosition objects
    
    New Methods:
    -----------
    get_unwrapped : unwrap angular position across 2π boundaries
    get_run_direction_epochs : detect clockwise/counterclockwise runs
    """
    
    def __init__(self, *args, radius=91.44, **kwargs):
        """
        Initialize CircularPosition object.
        
        Parameters
        ----------
        *args : 
            Positional arguments passed to parent Position class
        radius : float, optional
            Radius of the circular track in cm. If provided, this can be used
            for distance calculations and track geometry operations.
            Default: 91.44
        **kwargs : 
            Keyword arguments passed to parent Position class
            
        Raises
        ------
        AssertionError
            If traces are not 1-dimensional
            
        Examples
        --------
        >>> # Initialize with radius
        >>> pos = CircularPosition(traces=theta, sampling_rate=30, 
        ...                        t_start=0, radius=91.44)
        
        >>> # Initialize without radius (can set later)
        >>> pos = CircularPosition(traces=theta, sampling_rate=30, t_start=0)
        >>> pos.radius = 91.44
        """
        super().__init__(*args, **kwargs)
        
        # Require 1D position data
        assert self.ndim == 1, "CircularPosition requires 1-dimensional position data"
        
        self._radius = radius
    
    # ==================== radius Properties ====================

    @property
    def radius(self):
        """
        Get the radius of the circular track.
        
        Returns
        -------
        float or None
            Track radius in cm, or None if not set
        """
        return self._radius
    
    
    @radius.setter
    def radius(self, value):
        """
        Set the radius of the circular track.
        
        Parameters
        ----------
        value : float or None
            Track radius in cm
        """
        if value is not None and value <= 0:
            raise ValueError("Radius must be positive")
        self._radius = value


    # ==================== Velocity & Speed Properties ====================
    
    @property
    def velocity(self):
        """
        Calculate linearized velocity from unwrapped angular position.
        
        Unwraps the angular position to handle 2π boundaries, then computes
        velocity as the derivative of the unwrapped position. This preserves
        direction (sign) and handles circular wrapping correctly.
        
        Returns
        -------
        np.ndarray
            Linearized velocity accounting for circular wrapping (preserves sign)
        """
        dt = 1 / self.sampling_rate
        
        # Unwrap angular position to handle 2π boundaries
        unwrapped = np.unwrap(self.traces[0])
        
        # Calculate velocity from unwrapped position
        velocity = np.diff(unwrapped) / dt
        
        # Prepend 0 for first sample (no previous position)
        return np.hstack(([0], velocity))
    
    
    @property
    def speed(self):
        """
        Calculate linearized speed as the magnitude of velocity.
        
        Speed is computed as the absolute value of linearized velocity,
        accounting for circular wrapping.
        
        Returns
        -------
        np.ndarray
            Speed at each time point (magnitude, always positive)
        """
        return np.abs(self.velocity)


    # ==================== Time Slicing ====================
    
    def time_slice(self, t_start, t_stop, zero_times=False):
        """
        Extract a temporal slice of the position data.
        
        Parameters
        ----------
        t_start : float
            Start time in seconds
        t_stop : float
            Stop time in seconds
        zero_times : bool, optional
            If True, reset times to start at 0 (default: False)
            
        Returns
        -------
        CircularPosition
            New CircularPosition object with sliced data
        """
        indices = super()._time_slice_params(t_start, t_stop)
        
        if zero_times:
            t_stop = t_stop - t_start
            t_start = 0
        
        if self.traces_rot is not None:
            return CircularPosition(
                traces=self.traces[:, indices],
                traces_rot=self.traces_rot[:, indices],
                sampling_rate=self.sampling_rate,
                t_start=t_start,
                radius=self.radius,
            )
        else:
            return CircularPosition(
                traces=self.traces[:, indices],
                t_start=t_start,
                sampling_rate=self.sampling_rate,
                radius=self.radius,
            )


    # ==================== Run Direction Analysis ====================
    
    def get_run_direction_epochs(self, speed_thresh=(0.3 * 91.44, None), speed_2D=None,
                         edge_cutoff=np.pi/15 * 91.44,
                         sep=0.5, min_distance=np.pi/30 * 91.44, sigma_t=0.05):
        """
        Detect and classify running epochs by direction (clockwise/counterclockwise).
        
        LOGIC UPDATE: Neighbors are merged based on 'sep' FIRST, then filtered by 'min_distance'.
        """
        # Store parameters as metadata
        metadata = {
            'speed_thresh': speed_thresh,
            'speed_2D': speed_2D if speed_2D is None else 'provided',
            'edge_cutoff': edge_cutoff,
            'sep': sep,
            'min_distance': min_distance,
            'sigma_t': sigma_t
        }
        
        # Convert speed threshold to radians/s
        speed_thresh_rad = (
            speed_thresh[0] / self.radius if speed_thresh[0] is not None else None,
            speed_thresh[1] / self.radius if speed_thresh[1] is not None and not np.isinf(speed_thresh[1]) else speed_thresh[1]
        )
        edge_cutoff_rad = edge_cutoff / self.radius
        min_distance_rad = min_distance / self.radius
        
        sampling_rate = self.sampling_rate
        dt = 1 / sampling_rate
        
        # Get unwrapped position data
        x = self.get_unwrapped()
        
        # Calculate or use provided speed
        if speed_2D is not None:
            speed = speed_2D
        else:
            if sigma_t <= 0:
                speed = np.abs(self.velocity)
            else:
                speed = gaussian_filter1d(self.velocity, sigma=sigma_t / dt)
                speed = np.abs(speed)
        
        # 1. Detect raw high-speed epochs
        epochs = detect_epochs_peak(
            signal=speed, 
            edge_cutoff=edge_cutoff_rad, 
            lowthresh=speed_thresh_rad[0], 
            highthresh=speed_thresh_rad[-1], 
            prominence=0
        )
        
        # Extract just start and stop indices (peaks will be recalculated after merge)
        if epochs.size > 0:
            raw_indices = epochs[:, :2].astype(int)
        else:
            raw_indices = np.empty((0, 2), dtype=int)

        # 2. Merge Neighbors First
        merged_indices = []
        if len(raw_indices) > 0:
            if sep is not None and sep > 0:
                sep_samples = int(sep * sampling_rate)
                
                # Start with the first epoch
                curr_start, curr_stop = raw_indices[0]
                
                for next_start, next_stop in raw_indices[1:]:
                    # If gap is smaller than separation, merge
                    if (next_start - curr_stop) <= sep_samples:
                        curr_stop = next_stop
                    else:
                        # Gap is large, save current and start new
                        merged_indices.append([curr_start, curr_stop])
                        curr_start, curr_stop = next_start, next_stop
                
                # Append the final epoch
                merged_indices.append([curr_start, curr_stop])
                merged_indices = np.array(merged_indices)
            else:
                merged_indices = raw_indices
        else:
            merged_indices = np.empty((0, 2), dtype=int)

        # 3. Filter by Distance & Determine Direction
        # We construct lists to build the DataFrame later
        final_starts = []
        final_stops = []
        final_labels = []
        final_peak_times = []
        final_peak_speeds = []

        for epoch in merged_indices:
            start_idx, stop_idx = epoch
            
            # Calculate displacement on the MERGED epoch
            displacement = x[stop_idx] - x[start_idx]
            
            # Check min_distance
            if np.abs(displacement) > min_distance_rad:
                direction = "counter" if displacement > 0 else "clockwise"
                
                # Recalculate peak for the new merged range
                # Find index of max speed within this window (relative to window)
                epoch_speed_slice = speed[start_idx:stop_idx]
                if len(epoch_speed_slice) > 0:
                    rel_peak_idx = np.argmax(epoch_speed_slice)
                    abs_peak_idx = start_idx + rel_peak_idx
                    peak_val = epoch_speed_slice[rel_peak_idx]
                else:
                    # Fallback for edge cases
                    abs_peak_idx = start_idx
                    peak_val = 0
                
                final_starts.append(start_idx)
                final_stops.append(stop_idx)
                final_labels.append(direction)
                final_peak_times.append(abs_peak_idx)
                final_peak_speeds.append(peak_val)

        # 4. Create Final DataFrame
        if len(final_starts) > 0:
            high_speed_time = np.vstack((final_starts, final_stops)).T / sampling_rate + self.t_start
            high_speed_time = np.around(high_speed_time, 2)
            
            peak_time_arr = np.array(final_peak_times) / sampling_rate + self.t_start
            
            run_events = pd.DataFrame(high_speed_time, columns=["start", "stop"])
            run_events["label"] = final_labels
            run_events["peak_time"] = peak_time_arr
            run_events["peak_speed"] = final_peak_speeds
        else:
            # Return empty DataFrame structure if no epochs found
            run_events = pd.DataFrame(columns=["start", "stop", "label", "peak_time", "peak_speed"])

        run_epochs = Epoch(epochs=run_events, metadata=metadata)
        
        return run_epochs

    
    def get_unwrapped(self):
        """
        Unwrap 1D circular position across 2π boundaries.
        
        Uses numpy.unwrap to detect jumps across the 2π boundary and 
        accumulate corrections to produce a continuous unwrapped position trace.
        
        Returns
        -------
        np.ndarray
            Unwrapped position (can exceed [0, 2π] range)
        """
        return np.unwrap(self.x)