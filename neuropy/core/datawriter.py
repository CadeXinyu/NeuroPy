import numpy as np
import pandas as pd
from pathlib import Path
import datetime
from typing import Optional, Union, Any, Dict


class DataWriter:
    """
    Base class providing data persistence and metadata management.
    
    This class should be inherited by data classes that need:
    - Save/load functionality for numpy files
    - Standardized metadata handling
    - Time-based slicing utilities
    
    Attributes:
        _metadata: Dictionary storing arbitrary metadata about the data object
    """
    
    def __init__(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize DataWriter with optional metadata.
        
        Args:
            metadata: Dictionary containing metadata about the data object.
                     If None, an empty dictionary is initialized.
        
        Raises:
            AssertionError: If metadata is not a dictionary
        """
        if metadata is not None:
            assert isinstance(metadata, dict), "Only dictionary accepted as metadata"
            self._metadata: Dict[str, Any] = metadata
        else:
            self._metadata: Dict[str, Any] = {}
    
    # ==================== Metadata Management ====================
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the metadata dictionary."""
        return self._metadata
    
    @metadata.setter
    def metadata(self, new_metadata: Optional[Dict[str, Any]]) -> None:
        """
        Update metadata by merging with existing metadata.
        
        Args:
            new_metadata: Dictionary to merge into existing metadata.
                         If None, no update is performed.
        
        Raises:
            AssertionError: If new_metadata is not a dictionary
        """
        if new_metadata is not None:
            assert isinstance(new_metadata, dict), "Only dictionary accepted"
            self._metadata = self._metadata | new_metadata
    
    # ==================== Serialization ====================
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert object attributes to a dictionary for serialization.
        
        Converts all object attributes to a dictionary, with special handling for:
        - Private attributes (removes leading underscore)
        - pandas DataFrames (converts to dict to avoid pickling issues)
        
        Returns:
            Dictionary containing all object attributes
        """
        result = {}
        
        for attr_name in self.__dict__.keys():
            attr_value = getattr(self, attr_name)
            
            # Convert pandas DataFrames to dict to avoid pickling errors
            if isinstance(attr_value, pd.DataFrame):
                attr_value = attr_value.to_dict()
            
            # Remove leading underscore from private attributes
            clean_name = attr_name[1:] if attr_name.startswith("_") else attr_name
            result[clean_name] = attr_value
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataWriter':
        """
        Create an instance from a dictionary.
        
        Args:
            data: Dictionary containing object attributes
        
        Returns:
            New instance of the class initialized with the dictionary data
        """
        return cls(**data)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save object to disk as a numpy .npy file.
        
        Args:
            filepath: Path where the file should be saved
        
        Raises:
            AssertionError: If filepath is not a string or Path object
        """
        assert isinstance(filepath, (str, Path)), "filename is invalid"
        
        data = self.to_dict()
        np.save(filepath, data)
        print(f"{filepath} saved")
    
    def save_with_date(self, filepath: Union[str, Path]) -> None:
        """
        Save object with current date appended to filename.
        
        Appends date in format .DD-MM-YY to the filename before saving.
        Example: 'data.npy' becomes 'data.npy.18-10-25'
        
        Args:
            filepath: Base path where the file should be saved
        
        Raises:
            AssertionError: If filepath is not a string or Path object
        """
        assert isinstance(filepath, (str, Path)), "filename is invalid"
        
        # Convert to Path object if needed
        filepath = Path(filepath) if isinstance(filepath, str) else filepath
        
        # Create date suffix
        date_suffix = "." + datetime.date.today().strftime("%d-%m-%y")
        filename_with_date = filepath.name + date_suffix
        
        # Update filepath with new name
        filepath = filepath.with_name(filename_with_date)
        
        # Save the data
        data = self.to_dict()
        np.save(filepath, data)
        print(f"{filepath} saved")
    
    @classmethod
    def from_file(
        cls,
        filepath: Union[str, Path],
        convert: bool = False
    ) -> Optional[Union['DataWriter', Dict[str, Any]]]:
        """
        Load object from a saved numpy .npy file.
        
        Args:
            filepath: Path to the file to load
            convert: If True, returns a class instance. If False, returns raw dict.
                    Default is False for legacy compatibility.
        
        Returns:
            Either a class instance (if convert=True) or dictionary (if convert=False).
            Returns None if file doesn't exist.
        """
        # Convert to Path object
        filepath = Path(filepath) if isinstance(filepath, str) else filepath
        
        if not filepath.is_file():
            return None
        
        # Load the data
        data = np.load(filepath, allow_pickle=True).item()
        
        # Convert to class instance if requested
        if convert:
            data = cls.from_dict(data)
        
        return data
    
    # ==================== Time Slicing Utilities ====================
    
    def _time_slice_params(
        self,
        t1: Optional[float] = None,
        t2: Optional[float] = None
    ) -> Union[np.ndarray, tuple]:
        """
        Generate parameters for time-based slicing.
        
        This helper method is used by subclasses to implement time slicing.
        It returns either:
        - Boolean array for indexing (if object has 'time' attribute)
        - Tuple of (t1, t2) values (otherwise)
        
        Args:
            t1: Start time. If None, uses object's t_start
            t2: End time. If None, uses object's t_stop
        
        Returns:
            Either a boolean numpy array for indexing, or a tuple of (t1, t2)
        
        Raises:
            AssertionError: If t2 <= t1
        """
        # Use object's default times if not provided
        if t1 is None:
            t1 = self.t_start
        
        if t2 is None:
            t2 = self.t_stop
        
        assert t2 > t1, "t2 must be greater than t1"
        
        # Return boolean index if time array exists, otherwise return time values
        if hasattr(self, "time"):
            return (self.time >= t1) & (self.time <= t2)
        else:
            return t1, t2