from pathlib import Path
import os
from ..io import neuroscopeio
from ..io import binarysignalio
import neuropy.core as core

class ProcessData:
    """
    A class to load and manage electrophysiology data files.
    
    This class handles loading of XML metadata, probe configuration, 
    EEG signals, and raw DAT files from a recording session directory.
    
    Attributes
    ----------
    basepath : Path
        Directory containing the recording files
    filePrefix : Path
        Common file prefix (derived from XML filename without extension)
    probegroup : ProbeGroup
        Probe configuration loaded from .probegroup.npy file
    recinfo : NeuroscopeIO
        Recording metadata loaded from XML file
    eegfile : BinarysignalIO, optional
        EEG signal data (if .eeg file exists)
    datfile : BinarysignalIO, optional
        Raw signal data (if .dat file exists)
    """
    
    def __init__(self, basepath=os.getcwd()):
        """
        Initialize ProcessData by loading all available recording files.
        
        Parameters
        ----------
        basepath : str or Path, optional
            Path to directory containing recording files. Defaults to current directory.
        """
        # Convert to Path object for easier file manipulation
        basepath = Path(basepath)
        self.basepath = basepath
        
        # Find and validate XML file (contains recording metadata)
        xml_files = sorted(basepath.glob("*.xml"))
        assert len(xml_files) == 1, "Found fewer/more than one .xml file"
        
        # Extract file prefix (used for naming other files)
        file_prefix = xml_files[0].with_suffix("")
        self.filePrefix = file_prefix
        
        # Load probe configuration
        self.probegroup = core.ProbeGroup.from_file(
            file_prefix.with_suffix(".probegroup.npy")
        )
        
        # Load recording metadata from XML
        self.recinfo = neuroscopeio.NeuroscopeIO(xml_files[0])
        
        # Attempt to load EEG file (downsampled LFP data)
        eeg_files = sorted(basepath.glob("*.eeg"))
        try:
            assert len(eeg_files) == 1, "Fewer/more than one .eeg file detected"
            self.eegfile = binarysignalio.BinarysignalIO(
                eeg_files[0],
                n_channels=self.recinfo.n_channels,
                sampling_rate=self.recinfo.eeg_sampling_rate,
            )
        except AssertionError:
            self.eegfile = None
            print("Fewer/more than one .eeg file detected, no EEG file loaded")
        
        # Attempt to load DAT file (raw high-sampling-rate data)
        try:
            dat_file_path = eeg_files[0].with_suffix(".dat")
            self.datfile = binarysignalio.BinarysignalIO(
                dat_file_path,
                n_channels=self.recinfo.n_channels,
                sampling_rate=self.recinfo.dat_sampling_rate,
            )
        except (FileNotFoundError, IndexError):
            self.datfile = None
            print("No DAT file found, not loading")
    
    def __repr__(self) -> str:
        """
        Return detailed string representation of the ProcessData object.
        
        Returns
        -------
        str
            Multi-line string showing:
            - Class name and XML filename
            - Number of channels and shanks
            - Sampling rates (if files loaded)
            - File availability status
        """
        lines = [
            f"{self.__class__.__name__}('{self.recinfo.source_file.name}')",
            f"  Base path: {self.basepath}",
            f"  Channels: {self.recinfo.n_channels}",
            f"  Shanks: {self.probegroup.n_shanks}",
        ]
        
        # Add EEG info only if loaded
        if self.eegfile is not None:
            lines.append(f"  EEG sampling rate: {self.recinfo.eeg_sampling_rate} Hz")
            lines.append(f"  EEG file: ✓ loaded")
        else:
            lines.append(f"  EEG file: ✗ not loaded")
        
        # Add DAT info only if loaded
        if self.datfile is not None:
            lines.append(f"  DAT sampling rate: {self.recinfo.dat_sampling_rate} Hz")
            lines.append(f"  DAT file: ✓ loaded")
        else:
            lines.append(f"  DAT file: ✗ not loaded")
        
        return '\n'.join(lines)