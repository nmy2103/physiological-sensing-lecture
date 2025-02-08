from .ECG import Filters as ECGFilters
from .PPG import Filters as PPGFilters
from .ECG import BeatDetectors as ECGBeatDetectors
from .PPG import BeatDetectors as PPGBeatDetectors
from .SQA import Cardio as cardio_sqa
from typing import Literal, Union
from numpy import ndarray

def filter_ecg(
    signal: Union[ndarray, list], 
    fs: int, 
    lowcut: float = 1.0, 
    highcut: float = 15.0,
    order: int = 2,
) -> ndarray:
    """
    Remove baseline drift, EMG noise, and powerline interference from an 
    ECG signal with an elliptic bandpass filter.
    
    Parameters
    ---------
    signal : array-like
        An array containing the input ECG signal to be filtered.
    fs : int
        The sampling frequency, in Hz, of the ECG signal.
    lowcut : float, optional
        The lower cutoff frequency of the bandpass filter; by default, 1.0 Hz.
    highcut : float, optional
        The upper cutoff frequency of the bandpass filter; by default, 15.0 Hz.
    order : int, optional
        The order of the filter, controlling its sharpness; by default, 2.

    Returns
    -------
    array-like
        An array containing the filtered ECG signal.
    """
    return ECGFilters(fs = fs).filter_signal(
        signal = signal, 
        lowcut = lowcut, 
        highcut = highcut, 
        order = order)

def filter_ppg(
    signal: Union[ndarray, list], 
    fs: int,
    lowcut: float = 0.5, 
    highcut: float = 10, 
    order: int = 4, 
    window_len: int = 0.5
) -> ndarray:
    """
    Remove baseline drift, EMG noise, and powerline interference from a 
    PPG signal with Chebyshev Type II and moving average filters.

    Parameters
    ----------
    signal : array-like
        An array containing the input PPG signal to be filtered.
    fs : int
        The sampling frequency, in Hz, of the PPG signal.
    lowcut : int, float
        The cut-off frequency at which frequencies below this value in the
        signal are attenuated; by default, 0.5 Hz.
    highcut : int, float
        The cut-off frequency at which frequencies above this value in the
        signal are attenuated; by default, 10 Hz.
    order : int
        The filter order, i.e., the number of samples required to
        produce the desired filtered output; by default, 4.
    window_len : int
        The size of the moving average window, in seconds.

    Returns
    -------
    array-like
        An array containing the filtered PPG data.
    """
    return PPGFilters(fs = fs).filter_signal(
        signal = signal,
        lowcut = lowcut,
        highcut = highcut,
        order = order, 
        window_len = window_len
    )

__all__ = ['filter_ecg', 
           'filter_ppg', 
           'ECGBeatDetectors', 
           'PPGBeatDetectors', 
           'cardio_sqa']