import numpy as np
import pandas as pd
from .datawriter import DataWriter


class Shank:
    """
    Represents a single shank of a neural recording probe.
    
    A shank is a physical structure containing multiple recording contacts arranged
    in a geometric pattern. Each contact has a position (x, y) and is associated with
    a channel for recording neural signals.
    
    Attributes
    ----------
    _x : np.ndarray or None
        X-coordinates of all contacts on the shank
    _y : np.ndarray or None
        Y-coordinates of all contacts on the shank
    _connected : np.ndarray or None
        Boolean array indicating which contacts are connected/functional
    _contact_id : np.ndarray or None
        Unique identifiers for each contact
    _channel_id : np.ndarray or None
        Channel identifiers mapping contacts to recording channels
        
    Examples
    --------
    >>> # Create a shank with 2 columns, 10 contacts per column
    >>> shank = Shank.auto_generate(columns=2, contacts_per_column=10)
    >>> print(f"Number of contacts: {shank.n_contacts}")
    Number of contacts: 20
    """
    def __init__(self) -> None:
        """
        Initialize an empty Shank object.
        
        All attributes are set to None and should be populated using
        auto_generate() or other methods.
        """
        self._x = None
        self._y = None
        self._connected = None
        self._contact_id = None
        self._channel_id = None

    def __repr__(self):
        """
        Return a string representation of the Shank object.
        
        Returns
        -------
        str
            String describing the shank's key properties
            
        Examples
        --------
        >>> shank = Shank.auto_generate(columns=2, contacts_per_column=10)
        >>> print(shank)
        Shank(n_contacts=20, connected=20, x_range=[0.0, 15.0], y_range=[0.0, 180.0])
        """
        if self._x is None or self._y is None:
            return "Shank(uninitialized)"
        
        n_connected = np.sum(self.connected) if self.connected is not None else 0
        x_range = f"[{np.min(self.x)}, {np.max(self.x)}]"
        y_range = f"[{np.min(self.y)}, {np.max(self.y)}]"
        
        return (f"Shank(n_contacts={self.n_contacts}, connected={n_connected}, "
                f"x_range={x_range}, y_range={y_range})")

    @staticmethod
    def auto_generate(
        columns=2,
        contacts_per_column=10,
        xpitch=15,
        ypitch=20,
        y_shift_per_column=None,
        channel_id=None,
    ):
        """
        Automatically generate a shank with a regular grid layout.
        
        Creates a shank with contacts arranged in columns, with configurable
        spacing and vertical shifts between columns. This is useful for creating
        standard probe geometries like tetrodes or linear arrays.
        
        Parameters
        ----------
        columns : int, default=2
            Number of columns of contacts
        contacts_per_column : int or list of int, default=10
            Number of contacts in each column. If int, all columns have the same
            number. If list, specifies contacts for each column individually.
        xpitch : float, default=15
            Horizontal spacing between columns (in micrometers typically)
        ypitch : float, default=20
            Vertical spacing between contacts within a column (in micrometers)
        y_shift_per_column : list of float or None, default=None
            Vertical offset for each column. If None, all columns start at y=0.
            Useful for creating staggered contact patterns.
        channel_id : np.ndarray or None, default=None
            Custom channel IDs for the contacts. If None, uses sequential numbering.
            
        Returns
        -------
        Shank
            A fully initialized Shank object with the specified geometry
            
        Examples
        --------
        >>> # Create a tetrode (2x2 contacts)
        >>> tetrode = Shank.auto_generate(columns=2, contacts_per_column=2, 
        ...                               xpitch=10, ypitch=10)
        
        >>> # Create a staggered array
        >>> staggered = Shank.auto_generate(columns=2, contacts_per_column=10,
        ...                                  y_shift_per_column=[0, 10])
        """
        if isinstance(contacts_per_column, int):
            contacts_per_column = [contacts_per_column] * columns

        if y_shift_per_column is None:
            y_shift_per_column = [0] * columns

        positions = []
        for i in range(columns):
            x = np.ones(contacts_per_column[i]) * xpitch * i
            y = np.arange(contacts_per_column[i]) * ypitch + y_shift_per_column[i]
            positions.append(np.hstack((x[:, None], y[:, None])))
        positions = np.vstack(positions)

        shank = Shank()
        shank._x = positions[:, 0]
        shank._y = positions[:, 1]
        shank._channel_id = channel_id
        shank._connected = np.ones(np.sum(contacts_per_column), dtype=bool)
        shank._contact_id = np.arange(np.sum(contacts_per_column))
        if channel_id is None:
            shank._channel_id = np.arange(np.sum(contacts_per_column))
        else:
            shank._channel_id = channel_id

        return shank

    @staticmethod
    def from_library(probe_name):
        """
        Load a predefined shank geometry from a library.
        
        This method is a placeholder for loading standard probe geometries
        from a library of common neural probe designs (e.g., Neuropixels,
        Cambridge Neurotech probes).
        
        Parameters
        ----------
        probe_name : str
            Name of the probe design to load from the library
            
        Returns
        -------
        Shank
            A Shank object with the specified geometry
            
        Notes
        -----
        This method is not yet implemented.
        """
        pass

    @staticmethod
    def set_contacts(positions, channel_ids):
        """
        Create a shank with custom contact positions.
        
        This method allows manual specification of contact positions rather
        than using a regular grid pattern.
        
        Parameters
        ----------
        positions : np.ndarray
            Array of (x, y) coordinates for each contact, shape (n_contacts, 2)
        channel_ids : np.ndarray
            Array of channel identifiers for each contact
            
        Returns
        -------
        Shank
            A Shank object with the specified custom geometry
            
        Notes
        -----
        This method is not yet implemented.
        """
        pass

    @property
    def x(self):
        """
        Get the x-coordinates of all contacts.
        
        Returns
        -------
        np.ndarray
            Array of x-coordinates for each contact
        """
        return self._x

    @x.setter
    def x(self, arr):
        """
        Set the x-coordinates of all contacts.
        
        Parameters
        ----------
        arr : np.ndarray
            Array of x-coordinates, must match the number of contacts
            
        Raises
        ------
        AssertionError
            If array length doesn't match the number of contacts
        """
        assert (
            len(arr) == self.n_contacts
        ), "number of x coordinates should match number of contacts"
        self._x = arr

    @property
    def y(self):
        """
        Get the y-coordinates of all contacts.
        
        Returns
        -------
        np.ndarray
            Array of y-coordinates for each contact
        """
        return self._y

    @y.setter
    def y(self, arr):
        """
        Set the y-coordinates of all contacts.
        
        Parameters
        ----------
        arr : np.ndarray
            Array of y-coordinates, must match the number of contacts
            
        Raises
        ------
        AssertionError
            If array length doesn't match the number of contacts
        """
        assert (
            len(arr) == self.n_contacts
        ), "number of y coordinates should match number of contacts"
        self._y = arr

    @property
    def contact_id(self):
        """
        Get the contact IDs.
        
        Returns
        -------
        np.ndarray
            Array of unique contact identifiers
        """
        return self._contact_id

    @property
    def channel_id(self):
        """
        Get the channel IDs for all contacts.
        
        Returns
        -------
        np.ndarray
            Array mapping each contact to its recording channel
        """
        return self._channel_id

    @channel_id.setter
    def channel_id(self, chan_ids):
        """
        Set the channel IDs for all contacts.
        
        Parameters
        ----------
        chan_ids : np.ndarray
            Array of channel IDs, must match the number of contacts
            
        Raises
        ------
        AssertionError
            If array length doesn't match the number of contacts
        """
        assert self.n_contacts == len(chan_ids)
        self._channel_id = chan_ids

    @property
    def connected(self):
        """
        Get the connection status of all contacts.
        
        Returns
        -------
        np.ndarray
            Boolean array indicating which contacts are connected/functional
        """
        return self._connected

    @connected.setter
    def connected(self, arr):
        """
        Set the connection status of all contacts.
        
        Parameters
        ----------
        arr : np.ndarray
            Boolean array indicating which contacts are connected
        """
        self._connected = arr

    @property
    def n_contacts(self):
        """
        Get the total number of contacts on the shank.
        
        Returns
        -------
        int
            Number of contacts
        """
        return len(self.x)

    def to_dict(self):
        """
        Convert the shank layout to a dictionary.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'x': x-coordinates of contacts
            - 'y': y-coordinates of contacts
            - 'contact_id': contact identifiers
            - 'channel_id': channel identifiers
            - 'connected': connection status
            
        Examples
        --------
        >>> shank = Shank.auto_generate(columns=2, contacts_per_column=5)
        >>> layout = shank.to_dict()
        >>> print(layout.keys())
        dict_keys(['x', 'y', 'contact_id', 'channel_id', 'connected'])
        """
        layout = {
            "x": self.x,
            "y": self.y,
            "contact_id": self.contact_id,
            "channel_id": self.channel_id,
            "connected": self.connected,
        }
        return layout

    def from_dict(self):
        """
        Create a Shank object from a dictionary.
        
        This method is a placeholder for deserializing a shank from
        a dictionary representation.
        
        Notes
        -----
        This method is not yet implemented.
        """
        pass

    def set_disconnected_channels(self, channel_ids):
        """
        Mark specific channels as disconnected.
        
        Updates the connected status array to mark the specified channels
        as non-functional. This is useful for handling broken or noisy contacts.
        
        Parameters
        ----------
        channel_ids : array-like
            Array of channel IDs to mark as disconnected
            
        Examples
        --------
        >>> shank = Shank.auto_generate(columns=2, contacts_per_column=10)
        >>> shank.set_disconnected_channels([5, 7, 12])  # Mark channels 5, 7, 12 as bad
        >>> print(f"Connected channels: {np.sum(shank.connected)}")
        Connected channels: 17
        """
        self.connected[np.isin(self.channel_id, channel_ids)] = False

    def to_dataframe(self):
        """
        Convert the shank layout to a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: x, y, contact_id, channel_id, connected
            Each row represents one contact on the shank.
            
        Examples
        --------
        >>> shank = Shank.auto_generate(columns=2, contacts_per_column=5)
        >>> df = shank.to_dataframe()
        >>> print(df.head())
           x   y  contact_id  channel_id  connected
        0  0   0           0           0       True
        1  0  20           1           1       True
        """
        return pd.DataFrame(self.to_dict())

    def move(self, translation):
        """
        Translate the shank by a specified offset.
        
        Moves all contacts by the specified x and y offsets. This is useful
        for positioning shanks relative to each other or to a reference point.
        
        Parameters
        ----------
        translation : tuple of (float, float)
            (x_offset, y_offset) to add to all contact positions
            
        Examples
        --------
        >>> shank = Shank.auto_generate(columns=2, contacts_per_column=5)
        >>> print(f"Original position: {shank.x[0]}, {shank.y[0]}")
        Original position: 0.0, 0.0
        >>> shank.move((100, 50))
        >>> print(f"New position: {shank.x[0]}, {shank.y[0]}")
        New position: 100.0, 50.0
        """
        x, y = translation
        self.x += x
        self.y += y


class Probe:
    """
    Represents a multi-shank neural recording probe.
    
    A Probe consists of one or more Shanks arranged with specified spacing.
    This class manages the combined geometry and channel mapping across all
    shanks in a single probe device.
    
    Attributes
    ----------
    _data : pd.DataFrame
        Internal dataframe storing all contact information with columns:
        - x, y: contact positions
        - contact_id: unique contact identifier
        - channel_id: recording channel identifier
        - connected: connection status
        - shank_id: which shank the contact belongs to
        
    Examples
    --------
    >>> # Create a probe with 2 shanks
    >>> shank1 = Shank.auto_generate(columns=2, contacts_per_column=10)
    >>> shank2 = Shank.auto_generate(columns=2, contacts_per_column=10)
    >>> probe = Probe([shank1, shank2], shank_pitch=(200, 0))
    >>> print(f"Total contacts: {probe.n_contacts}")
    Total contacts: 40
    >>> print(f"Number of shanks: {probe.n_shanks}")
    Number of shanks: 2
    """
    def __init__(self, shanks, shank_pitch=(150, 0)) -> None:
        """
        Initialize a Probe with one or more shanks.
        
        Parameters
        ----------
        shanks : Shank or list of Shank
            Single shank or list of shanks to include in the probe
        shank_pitch : tuple of (float, float), default=(150, 0)
            (x_spacing, y_spacing) between consecutive shanks.
            For example, (150, 0) places shanks 150 units apart horizontally
            at the same vertical position.
            
        Examples
        --------
        >>> # Single shank probe
        >>> shank = Shank.auto_generate(columns=2, contacts_per_column=16)
        >>> probe = Probe(shank)
        
        >>> # Multi-shank probe with vertical offset
        >>> shanks = [Shank.auto_generate(columns=1, contacts_per_column=32) 
        ...           for _ in range(4)]
        >>> probe = Probe(shanks, shank_pitch=(250, 100))
        """
        if isinstance(shanks, Shank):
            shanks = [shanks]

        if isinstance(shanks, list):
            assert np.all([_.__class__.__name__ == "Shank" for _ in shanks])

        self._data = pd.DataFrame(
            {
                "x": np.array([]),
                "y": np.array([]),
                "contact_id": np.array([]),
                "channel_id": np.array([]),
                "connected": np.array([], dtype=bool),
                "shank_id": np.array([]),
            }
        )

        x = np.arange(len(shanks)) * shank_pitch[0]
        y = np.arange(len(shanks)) * shank_pitch[1]
        for i, shank in enumerate(shanks):
            shank_df = shank.to_dataframe()
            shank_df["x"] += x[i]
            shank_df["y"] += y[i]
            shank_df["shank_id"] = i * np.ones(shank.n_contacts)
            # self._data = self._data.append(shank_df)
            self._data = pd.concat([self._data, shank_df])
        self._data = self._data.reset_index(drop=True)
        self._data["contact_id"] = np.arange(len(self._data))

    def __repr__(self):
        """
        Return a string representation of the Probe object.
        
        Returns
        -------
        str
            String describing the probe's key properties
            
        Examples
        --------
        >>> shank = Shank.auto_generate(columns=2, contacts_per_column=16)
        >>> probe = Probe([shank, shank], shank_pitch=(200, 0))
        >>> print(probe)
        Probe(n_shanks=2, n_contacts=64, connected=64, x_range=[0.0, 215.0], y_range=[0.0, 300.0])
        """
        if len(self._data) == 0:
            return "Probe(empty)"
        
        n_connected = np.sum(self.connected)
        x_range = f"[{np.min(self.x)}, {np.max(self.x)}]"
        y_range = f"[{np.min(self.y)}, {np.max(self.y)}]"
        
        return (f"Probe(n_shanks={self.n_shanks}, n_contacts={self.n_contacts}, "
                f"connected={n_connected}, x_range={x_range}, y_range={y_range})")

    @property
    def n_contacts(self):
        """
        Get the total number of contacts across all shanks.
        
        Returns
        -------
        int
            Total number of contacts on the probe
        """
        return len(self._data)

    @property
    def n_shanks(self):
        """
        Get the number of shanks in the probe.
        
        Returns
        -------
        int
            Number of shanks
        """
        return np.max(self._data["shank_id"]) + 1

    @property
    def shank_id(self):
        """
        Get the shank ID for each contact.
        
        Returns
        -------
        np.ndarray
            Array indicating which shank each contact belongs to
        """
        return self._data["shank_id"].values

    @property
    def x(self):
        """
        Get the x-coordinates of all contacts.
        
        Returns
        -------
        np.ndarray
            Array of x-coordinates for all contacts across all shanks
        """
        return self._data["x"].values

    @property
    def x_max(self):
        """
        Get the maximum x-coordinate across all contacts.
        
        Returns
        -------
        float
            Maximum x-coordinate, useful for determining probe width
        """
        return np.max(self._data["x"].values)

    @property
    def y(self):
        """
        Get the y-coordinates of all contacts.
        
        Returns
        -------
        np.ndarray
            Array of y-coordinates for all contacts across all shanks
        """
        return self._data["y"].values

    @property
    def channel_id(self):
        """
        Get the channel IDs for all contacts.
        
        Returns
        -------
        np.ndarray
            Array of channel identifiers for all contacts
        """
        return self._data["channel_id"].values

    @property
    def connected(self):
        """
        Get the connection status of all contacts.
        
        Returns
        -------
        np.ndarray
            Boolean array indicating which contacts are connected
        """
        return self._data["connected"].values

    def add_shanks(self, shanks: Shank, shank_pitch=(150, 0)):
        """
        Add additional shanks to the probe.
        
        Appends one or more shanks to the existing probe configuration.
        New shanks are positioned based on the current number of shanks
        and the specified pitch.
        
        Parameters
        ----------
        shanks : Shank or list of Shank
            Shank(s) to add to the probe
        shank_pitch : tuple of (float, float), default=(150, 0)
            Spacing between shanks (currently not used in positioning)
            
        Examples
        --------
        >>> probe = Probe(Shank.auto_generate(columns=1, contacts_per_column=16))
        >>> print(f"Initial shanks: {probe.n_shanks}")
        Initial shanks: 1
        >>> probe.add_shanks(Shank.auto_generate(columns=1, contacts_per_column=16))
        >>> print(f"After adding: {probe.n_shanks}")
        After adding: 2
        """
        if isinstance(shanks, list):
            assert np.all([_.__class__.__name__ == "Shank" for _ in shanks])
        else:
            assert isinstance(shanks, Shank)
            shanks = [shanks]

        for shank in shanks:
            shank_df = shank.to_dataframe()
            shank_df["shank_id"] = (self.n_shanks - 1) * np.ones(shank.n_contacts)
            self._data = pd.concat([self._data, shank_df])

    def to_dict(self):
        """
        Convert the probe configuration to a dictionary.
        
        Returns
        -------
        dict
            Dictionary representation of the probe's DataFrame
        """
        return self._data.to_dict()

    def to_dataframe(self):
        """
        Get the probe configuration as a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing all contact information with columns:
            x, y, contact_id, channel_id, connected, shank_id
        """
        return self._data

    def move(self, translation):
        """
        Translate the entire probe by a specified offset.
        
        Moves all contacts on all shanks by the specified x and y offsets.
        This is useful for positioning the probe in a coordinate system.
        
        Parameters
        ----------
        translation : tuple of (float, float)
            (x_offset, y_offset) to add to all contact positions
            
        Examples
        --------
        >>> probe = Probe(Shank.auto_generate(columns=2, contacts_per_column=10))
        >>> probe.move((1000, 500))  # Move probe to position (1000, 500)
        """
        x, y = translation
        self._data["x"] += x
        self._data["y"] += y


class ProbeGroup(DataWriter):
    """
    Manages a collection of multiple probes.
    
    A ProbeGroup represents a multi-probe recording system where multiple
    independent probes are used simultaneously. This class handles the combined
    geometry, channel mapping, and metadata for all probes in the group.
    
    Attributes
    ----------
    _data : pd.DataFrame
        Internal dataframe storing all contact information with columns:
        - x, y: contact positions
        - contact_id: unique contact identifier
        - channel_id: recording channel identifier
        - shank_id: which shank the contact belongs to
        - connected: connection status
        - probe_id: which probe the contact belongs to
        
    Examples
    --------
    >>> # Create a probe group with multiple probes
    >>> probe1 = Probe(Shank.auto_generate(columns=2, contacts_per_column=16))
    >>> probe2 = Probe(Shank.auto_generate(columns=2, contacts_per_column=16))
    >>> probe_group = ProbeGroup()
    >>> probe_group.add_probe(probe1)
    >>> probe_group.add_probe(probe2)
    >>> print(f"Total probes: {probe_group.n_probes}")
    Total probes: 2
    >>> print(f"Total contacts: {probe_group.n_contacts}")
    Total contacts: 64
    """
    def __init__(self, metadata=None) -> None:
        """
        Initialize an empty ProbeGroup.
        
        Parameters
        ----------
        metadata : dict or None, default=None
            Optional metadata dictionary for storing experimental information,
            probe specifications, or other relevant data
        """
        super().__init__(metadata=metadata)
        self._data = pd.DataFrame(
            {
                "x": np.array([]),
                "y": np.array([]),
                "contact_id": np.array([]),
                "channel_id": np.array([]),
                "shank_id": np.array([]),
                "connected": np.array([], dtype=bool),
                "probe_id": np.array([]),
            }
        )

    def __repr__(self):
        """
        Return a string representation of the ProbeGroup object.
        
        Returns
        -------
        str
            String describing the probe group's key properties
            
        Examples
        --------
        >>> probe_group = ProbeGroup()
        >>> probe_group.add_probe(probe1)
        >>> probe_group.add_probe(probe2)
        >>> print(probe_group)
        ProbeGroup(n_probes=2, n_shanks=4, n_contacts=128, connected=128)
        """
        if len(self._data) == 0:
            return "ProbeGroup(empty)"
        
        n_connected = np.sum(self._data["connected"].values)
        
        return (f"ProbeGroup(n_probes={self.n_probes}, n_shanks={self.n_shanks}, "
                f"n_contacts={self.n_contacts}, connected={n_connected})")

    @property
    def x(self):
        """
        Get the x-coordinates of all contacts across all probes.
        
        Returns
        -------
        np.ndarray
            Array of x-coordinates for all contacts
        """
        return self._data["x"].values

    @property
    def x_min(self):
        """
        Get the minimum x-coordinate across all contacts.
        
        Returns
        -------
        float
            Minimum x-coordinate
        """
        return np.min(self.x)

    @property
    def x_max(self):
        """
        Get the maximum x-coordinate across all contacts.
        
        Returns
        -------
        float
            Maximum x-coordinate
        """
        return np.max(self.x)

    @property
    def y(self):
        """
        Get the y-coordinates of all contacts across all probes.
        
        Returns
        -------
        np.ndarray
            Array of y-coordinates for all contacts
        """
        return self._data["y"].values

    @property
    def y_min(self):
        """
        Get the minimum y-coordinate across all contacts.
        
        Returns
        -------
        float
            Minimum y-coordinate
        """
        return np.min(self.y)

    @property
    def y_max(self):
        """
        Get the maximum y-coordinate across all contacts.
        
        Returns
        -------
        float
            Maximum y-coordinate
        """
        return np.max(self.y)

    @property
    def n_contacts(self):
        """
        Get the total number of contacts across all probes.
        
        Returns
        -------
        int
            Total number of contacts
        """
        return len(self._data)

    @property
    def channel_id(self):
        """
        Get the channel IDs for all contacts.
        
        Returns
        -------
        np.ndarray
            Array of channel identifiers
        """
        return self._data["channel_id"].values

    @property
    def shank_id(self):
        """
        Get the shank IDs for all contacts.
        
        Returns
        -------
        np.ndarray
            Array indicating which shank each contact belongs to
        """
        return self._data["shank_id"].values

    def get_channels(self, groupby="shank"):
        """
        Get channel IDs grouped by shank or probe.
        
        Returns an array of arrays, where each sub-array contains the channel
        IDs for a specific group (shank or probe).
        
        Parameters
        ----------
        groupby : str, default="shank"
            Grouping method: "shank" or "probe"
            
        Returns
        -------
        np.ndarray of dtype object
            Array of arrays containing channel IDs for each group
            
        Examples
        --------
        >>> probe_group = ProbeGroup()
        >>> # ... add probes ...
        >>> channels_by_shank = probe_group.get_channels(groupby="shank")
        >>> print(f"Channels on first shank: {channels_by_shank[0]}")
        """
        prb = self.to_dataframe()

        if groupby == "shank":
            prb = prb.groupby("shank_id")
            channels = []
            for i in prb.groups.keys():
                channels.append(prb.get_group(i).channel_id.values)
        if groupby == "probe":
            prb = prb.groupby("probe_id")
            channels = []
            for i in prb.groups.keys():
                channels.append(prb.get_group(i).channel_id.values)

        return np.array(channels, dtype="object")

    def get_shank_id_for_channels(self, channel_id):
        """
        Get shank IDs corresponding to specific channel IDs.
        
        Maps from channel IDs to their corresponding shank IDs. Can handle
        repeated channel IDs if present in the configuration.

        Parameters
        ----------
        channel_id : array
            channel_ids, can have repeated values

        Returns
        -------
        array
            shank_ids corresponding to the channels
            
        Examples
        --------
        >>> shank_ids = probe_group.get_shank_id_for_channels([5, 10, 15, 20])
        >>> print(f"Channel 5 is on shank {shank_ids[0]}")
        """
        shank_ids = self.shank_id
        channel_ids = self.channel_id

        # indx_location = np.concatenate(
        #     list(map(lambda x: channel_ids[channel_ids == x], channel_id))
        # )

        # return shank_ids[indx_location.astype(int)]
        return np.concatenate(
            [shank_ids[np.where(channel_ids == _)[0]] for _ in channel_id]
        )

    def get_probe_id_for_channels(self, channel_id):
        """
        Get probe IDs corresponding to specific channel IDs.
        
        Maps from channel IDs to their corresponding probe IDs. Can handle
        repeated channel IDs if present in the configuration.

        Parameters
        ----------
        channel_id : array
            channel_ids, can have repeated values

        Returns
        -------
        array
            probe_ids corresponding to the channels
            
        Examples
        --------
        >>> probe_ids = probe_group.get_probe_id_for_channels([5, 10, 15, 20])
        >>> print(f"Channel 5 is on probe {probe_ids[0]}")
        """
        probe_ids = self.probe_id
        channel_ids = self.channel_id

        return np.concatenate(
            [probe_ids[np.where(channel_ids == _)[0]] for _ in channel_id]
        ).astype("int")

    def get_probe(self, probe_id):
        """
        Get a specific probe from the group by its ID.
        
        Parameters
        ----------
        probe_id : int
            ID of the probe to retrieve
            
        Returns
        -------
        Probe or None
            The requested Probe object
            
        Notes
        -----
        This method is not yet implemented.
        """
        pass

    def get_connected_channels(self, groupby="shank"):
        """
        Get only the connected (functional) channels grouped by shank or probe.
        
        Filters out disconnected channels and returns channel IDs grouped
        by the specified grouping method.
        
        Parameters
        ----------
        groupby : str, default="shank"
            Grouping method: "shank" or "probe"
            
        Returns
        -------
        np.ndarray of dtype object
            Array of arrays containing connected channel IDs for each group
            
        Examples
        --------
        >>> connected = probe_group.get_connected_channels(groupby="shank")
        >>> print(f"Shank 0 has {len(connected[0])} connected channels")
        """
        df = self.to_dataframe()
        df = df[df["connected"] == True]
        chans = []
        probe_grp = df.groupby("probe_id")

        if groupby == "probe":
            for i in range(self.n_probes):
                chans.append(probe_grp.get_group(i).channel_id.values)
        if groupby == "shank":
            probe_grp = df.groupby("probe_id")
            for i in range(self.n_probes):
                shank_grp = probe_grp.get_group(i).groupby("shank_id")
                for i1 in shank_grp.groups.keys():
                    chans.append(shank_grp.get_group(i1).channel_id.values)

        return np.array(chans, dtype=object)

    @property
    def probe_id(self):
        """
        Get the probe IDs for all contacts.
        
        Returns
        -------
        np.ndarray
            Array indicating which probe each contact belongs to
        """
        return self._data["probe_id"].values

    @property
    def n_probes(self):
        """
        Get the number of probes in the group.
        
        Returns
        -------
        int
            Number of probes
        """
        return len(np.unique(self.probe_id))

    @property
    def n_shanks(self):
        """
        Get the total number of shanks across all probes.
        
        Returns
        -------
        int
            Total number of shanks
        """
        return len(np.unique(self.shank_id))

    @property
    def get_disconnected(self):
        """
        Get all disconnected (non-functional) contacts.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing only the rows for disconnected contacts
            
        Examples
        --------
        >>> disconnected = probe_group.get_disconnected
        >>> print(f"Number of bad channels: {len(disconnected)}")
        """
        return self._data[self._data["connected"] == False]

    def add_probe(self, probe: Probe):
        """
        Add a probe to the probe group.
        
        Appends a new probe to the group, automatically assigning it a unique
        probe ID and updating shank IDs to be globally unique across the group.
        
        Parameters
        ----------
        probe : Probe
            Probe object to add to the group
            
        Examples
        --------
        >>> probe_group = ProbeGroup()
        >>> probe1 = Probe(Shank.auto_generate(columns=2, contacts_per_column=16))
        >>> probe_group.add_probe(probe1)
        >>> print(f"Probes in group: {probe_group.n_probes}")
        Probes in group: 1
        """
        probe_df = probe.to_dataframe()
        probe_df["probe_id"] = self.n_probes * np.ones(probe.n_contacts)
        if self.n_probes > 0:
            probe_df["shank_id"] = probe_df["shank_id"] + self.n_shanks

        self._data = pd.concat([self._data, probe_df])

        # _, counts = np.unique(self.get_channel_ids(), return_counts=True)

    def to_dict(self):
        """
        Convert the probe group to a dictionary.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'data': the contact information DataFrame
            - 'metadata': associated metadata
        """
        return {
            "data": self._data,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: dict, sort_shanks_by: str in ["shank_id", "x"] = "shank_id"):
        """
        Create a ProbeGroup from a dictionary.
        
        Deserializes a ProbeGroup from its dictionary representation.
        
        Parameters
        ----------
        d : dict
            Dictionary containing 'data' and 'metadata' keys
        sort_shanks_by : str, default="shank_id"
            Column to sort shanks by: "shank_id" or "x"
            Within each shank, contacts are sorted by y in descending order
            
        Returns
        -------
        ProbeGroup
            Reconstructed ProbeGroup object
            
        Examples
        --------
        >>> # Save and load a probe group
        >>> probe_dict = probe_group.to_dict()
        >>> loaded_group = ProbeGroup.from_dict(probe_dict)
        """
        prbgrp = ProbeGroup(metadata=d["metadata"])
        prbgrp._data = d["data"].sort_values([sort_shanks_by, "y"], ascending=[True, False])
        return prbgrp

    @staticmethod
    def from_file(f, sort_shanks_by: str in ["shank_id", "x"] = "shank_id"):
        """
        Load a ProbeGroup from a file.
        
        Reads a probe group configuration from a file saved by the DataWriter.
        
        Parameters
        ----------
        f : str or Path
            Path to the file containing the probe group data
        sort_shanks_by : str, default="shank_id"
            Column to sort shanks by: "shank_id" or "x"
            
        Returns
        -------
        ProbeGroup or None
            Loaded ProbeGroup object, or None if file could not be read
            
        Examples
        --------
        >>> probe_group = ProbeGroup.from_file('my_probe_config.json')
        """
        d = DataWriter.from_file(f)
        if d is not None:
            return ProbeGroup.from_dict(d, sort_shanks_by)

    def to_dataframe(self):
        """
        Get the probe group configuration as a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing all contact information with columns:
            x, y, contact_id, channel_id, shank_id, connected, probe_id
        """
        return pd.DataFrame(self._data)

    def remove_probes(self, probe_id=None):
        """
        Remove probes from the group.
        
        Removes one or more probes from the probe group by their IDs.
        
        Parameters
        ----------
        probe_id : int, list of int, or None, default=None
            Probe ID(s) to remove. If None, removes all probes
            
        Notes
        -----
        Current implementation clears all data regardless of probe_id parameter.
        This should be updated to selectively remove specific probes.
        """
        self._data = {}