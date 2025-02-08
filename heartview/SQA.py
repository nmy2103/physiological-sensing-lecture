from typing import List, Literal, Tuple, Optional, Union
from tqdm import tqdm
from math import ceil, floor
from scipy.interpolate import interp1d
from scipy.signal import welch
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ============================== CARDIOVASCULAR ==============================
class Cardio:
    """
    A class for signal quality assessment on cardiovascular data, including
    electrocardiograph (ECG) or photoplethysmograph (PPG) data.

    Parameters/Attributes
    ---------------------
    fs : int
        The sampling rate of the cardiovascular data.
    """

    def __init__(
        self, 
        fs: int
    ) -> None:
        """
        Initialize the Cardio object.

        Parameters
        ----------
        fs : int
            The sampling rate of the ECG or PPG recording.
        """
        self.fs = int(fs)

    def compute_metrics(
        self, 
        data: pd.DataFrame, 
        beats_ix: Union[np.ndarray, list], 
        artifacts_ix: Union[np.ndarray, list], 
        ts_col: Optional[str] = None,
        seg_size: int = 60, 
        min_hr: Union[float, int] = 40, 
        rolling_window: Optional[int] = None,
        rolling_step: int = 15, 
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Compute all SQA metrics for cardiovascular data by segment or
        moving window. Metrics per segment or moving window include numbers
        of detected, expected, missing, and artifactual beats and
        percentages of missing and artifactual beats.

        Parameters
        ----------
        data : pandas.DataFrame
            A DataFrame containing pre-processed ECG or PPG data.
        beats_ix : array-like
            An array containing the indices of detected beats.
        artifacts_ix : array-like
            An array containing the indices of artifactual beats.
        seg_size : int
            The segment size in seconds; by default, 60.
        min_hr : int, float
            The minimum acceptable heart rate against which the number of
            beats in the last partial segment will be compared; by default, 40.
        ts_col : str, optional
            The name of the column containing timestamps; by default, None.
            If a string value is given, the output will contain a timestamps
            column.
        rolling_window : int, optional
            The size, in seconds, of the sliding window across which to
            compute the SQA metrics; by default, None.
        rolling_step : int, optional
            The step size, in seconds, of the sliding windows; by default, 15.
        show_progress : bool, optional
            Whether to display a progress bar while the function runs; by
            default, True.

        Returns
        -------
        metrics : pandas.DataFrame
            A DataFrame with all computed SQA metrics per segment.

        Notes
        -----
        If a value is given in the `rolling_window` parameter, the rolling
        window approach will override the segmented approach, ignoring any
        `seg_size` value.

        Examples
        --------
        >>> from heartview.pipeline import SQA
        >>> sqa = SQA.Cardio(fs = 1000)
        >>> artifacts_ix = sqa.identify_artifacts(beats_ix, method = 'both')
        >>> cardio_qa = sqa.compute_metrics(ecg, beats_ix, artifacts_ix, \
        ...                                 ts_col = 'Timestamp', \
        ...                                 seg_size = 60, min_hr = 40)
        """
        df = data.copy()
        df.index = df.index.astype(int)
        df.loc[beats_ix, 'Beat'] = 1

        if rolling_window is not None:
            metrics = pd.DataFrame()
            if ts_col is not None:
                seconds = self.get_seconds(data, beats_ix, ts_col,
                                           show_progress = show_progress)
                s = 1
                for n in tqdm(range(0, len(seconds), rolling_step),
                              disable = not show_progress):
        
                    # Get missing beats
                    window_missing = seconds.iloc[n:(n + rolling_window)]
                    # n_expected = window_missing['Mean HR'].median() * (seg_size / 60)
                    raw_expected = window_missing['Mean HR'].rolling(
                        window = seg_size + 1, min_periods = 1, center = True).median()
                    n_expected = raw_expected.median() * (seg_size / 60)
                    n_detected = window_missing['N Beats'].sum()
                    n_missing = (n_expected - n_detected) \
                        if n_expected > n_detected else 0
                    perc_missing = (n_missing / n_expected) * 100
                    ts = window_missing['Timestamp'].iloc[0]
        
                    # Summarize artifactual beats
                    artifacts = self.get_artifacts(
                        df, beats_ix, artifacts_ix, seg_size = 1,
                        ts_col = ts_col)
                    window_artifact = artifacts.iloc[n:(n + rolling_window)]
                    n_artifact = window_artifact['N Artifact'].sum()
                    perc_artifact = round((n_artifact / n_detected) * 100, 2)
        
                    # Output summary
                    metrics = pd.concat([metrics, pd.DataFrame.from_records([{
                        'Moving Window': s,
                        'Timestamp': ts,
                        'N Expected': int(n_expected),
                        'N Detected': int(n_detected),
                        'N Missing': int(n_missing),
                        '% Missing': round(perc_missing, 2),
                        'N Artifact': int(n_artifact),
                        '% Artifact': round(perc_artifact, 2)
                    }])], ignore_index = True).reset_index(drop = True)
                    s += 1
            else:
                seconds = self.get_seconds(data, beats_ix,
                                           show_progress = show_progress)
                s = 1
                for n in tqdm(range(0, len(seconds), rolling_step),
                              disable = not show_progress):

                    # Get missing beats
                    window_missing = seconds.iloc[n:(n + rolling_window)]
                    raw_expected = window_missing['Mean HR'].rolling(
                        window = seg_size + 1, min_periods = 1, center = True).median()
                    n_expected = raw_expected.median() * (seg_size / 60)
                    # n_expected = window_missing['Mean HR'].median() * (seg_size / 60)
                    n_detected = window_missing['N Beats'].sum()
                    n_missing = (n_expected - n_detected) \
                        if n_expected > n_detected else 0
                    perc_missing = (n_missing / n_expected) * 100

                    # Get artifactual beats
                    artifacts = self.get_artifacts(
                        df, beats_ix, artifacts_ix, seg_size = 1)
                    window_artifact = artifacts.iloc[n:(n + rolling_window)]
                    n_artifact = window_artifact['N Artifact'].sum()
                    perc_artifact = (n_artifact / n_detected) * 100

                    # Output summary
                    metrics = pd.concat([metrics, pd.DataFrame.from_records([{
                        'Moving Window': s,
                        'N Expected': int(n_expected),
                        'N Detected': int(n_detected),
                        'N Missing': int(n_missing),
                        '% Missing': round(perc_missing, 2),
                        'N Artifact': int(n_artifact),
                        '% Artifact': round(perc_artifact, 2)
                    }])], ignore_index = True).reset_index(drop = True)
                    s += 1

            # Handle last partial rolling window of data
            last_seg_len = len(seconds) % rolling_window
            if last_seg_len > 0:
                last_detected = metrics['N Detected'].iloc[-1]
                last_expected_ratio = min_hr / metrics['N Expected'].iloc[:-1].median()
                last_expected = last_expected_ratio * last_seg_len
                if last_expected > last_detected:
                    last_n_missing = last_expected - last_detected
                    last_perc_missing = (last_n_missing / last_expected) * 100
                else:
                    last_perc_missing = 0
                    last_n_missing = 0
                metrics['N Expected'].iloc[-1] = int(round(last_expected))
                metrics['N Missing'].iloc[-1] = int(round(last_n_missing))
                metrics['% Missing'].iloc[-1] = round(last_perc_missing, 2)

        else:
            if ts_col is not None:
                missing = self.get_missing(
                    df, beats_ix, seg_size, min_hr = min_hr, ts_col = ts_col,
                    show_progress = show_progress)
                artifacts = self.get_artifacts(
                    df, beats_ix, artifacts_ix, seg_size, ts_col)
                metrics = pd.merge(missing, artifacts,
                                   on = ['Segment', 'Timestamp'])
                metrics['Invalid'] = metrics['N Detected'].apply(
                    lambda n: 1 if n < int(min_hr * (seg_size/60)) or n > 220 else np.nan)
            else:
                missing = self.get_missing(
                    df, beats_ix, seg_size, show_progress = show_progress)
                artifacts = self.get_artifacts(
                    df, beats_ix, artifacts_ix, seg_size)
                metrics = pd.merge(missing, artifacts, on = ['Segment'])

        metrics['Invalid'] = metrics['N Detected'].apply(
            lambda x: 1 if x < int(min_hr * (seg_size/60)) or x > 220 else np.nan)

        return metrics

    def get_artifacts(
        self, 
        data: pd.DataFrame, 
        beats_ix: Union[np.ndarray, list], 
        artifacts_ix: Union[np.ndarray, list],
        seg_size: int = 60, 
        ts_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Summarize the number and proportion of artifactual beats per segment.

        Parameters
        ----------
        data : pandas.DataFrame
            A DataFrame containing the pre-processed ECG or PPG data.
        beats_ix : array-like
            An array containing the indices of detected beats.
        artifacts_ix : array-like
            An array containing the indices of artifactual beats. This is
            outputted from `SQA.Cardio.identify_artifacts()`.
        seg_size : int, optional
            The size of the segment in seconds; by default, 60.
        ts_col : str, optional
            The name of the column containing timestamps; by default, None.
            If a string value is given, the output will contain a timestamps
            column.

        Returns
        -------
        artifacts : pandas.DataFrame
            A DataFrame with the number and proportion of artifactual beats
            per segment.

        See Also
        --------
        SQA.Cardio.identify_artifacts :
            Identify artifactual beats using both or either of the methods.
        """
        df = data.copy()
        df.loc[beats_ix, 'Beat'] = 1
        df.loc[artifacts_ix, 'Artifact'] = 1

        n_seg = ceil(len(df) / (self.fs * seg_size))
        segments = pd.Series(np.arange(1, n_seg + 1))
        n_detected = df.groupby(
            df.index // (self.fs * seg_size))['Beat'].sum()
        n_artifact = df.groupby(
            df.index // (self.fs * seg_size))['Artifact'].sum()
        perc_artifact = (n_artifact / n_detected) * 100

        if ts_col is not None:
            timestamps = df.groupby(
                df.index // (self.fs * seg_size)).first()[ts_col]
            artifacts = pd.concat([
                segments,
                timestamps,
                n_artifact.astype(int),
                round(perc_artifact, 2),
            ], axis = 1)
            artifacts.columns = [
                'Segment',
                'Timestamp',
                'N Artifact',
                '% Artifact',
            ]
        else:
            artifacts = pd.concat([
                segments,
                n_artifact.astype(int),
                round(perc_artifact, 2),
            ], axis = 1)
            artifacts.columns = [
                'Segment',
                'N Artifact',
                '% Artifact',
            ]
        return artifacts

    def identify_artifacts(
        self,
        beats_ix: Union[np.ndarray, list], 
        method: Literal['hegarty', 'cbd', 'both'], 
        initial_hr: Optional[Union[int, float, Literal['auto']]] = None,
        prev_n: Optional[int] = None, 
        neighbors: Optional[int] = None, 
        tol: Optional[float] = None
    ) -> np.ndarray:
        """
        Identify locations of artifactual beats in cardiovascular data based
        on the criterion beat difference approach by Berntson et al. (1990),
        the Hegarty-Craver et al. (2018) approach, or both.

        Parameters
        ----------
        beats_ix : array-like
            An array containing the indices of detected beats.
        method : str
            The artifact identification method for identifying artifacts.
            This must be 'hegarty', 'cbd', or 'both'.
        initial_hr : int, float, or 'auto', optional
            The heart rate value for the first interbeat interval (IBI) to be
            validated against; by default, 'auto' for automatic calculation
            using the mean heart rate value obtained from six consecutive
            IBIs with the smallest average successive difference. Required
            for the 'hegarty' method.
        prev_n : int, optional
            The number of preceding IBIs to validate against; by default, 6.
            Required for 'hegarty' method.
        neighbors : int, optional
            The number of surrounding IBIs with which to derive the criterion
            beat difference score; by default, 5. Required for 'cbd' method.
        tol : float, optional
            A configurable hyperparameter used to fine-tune the stringency of
            the criterion beat difference test; by default, 1. Required for
            'cbd' method.

        Returns
        -------
        artifacts_ix : array-like
            An array containing the indices of identified artifact beats.

        Notes
        -----
        The source code for the criterion beat difference test is from work by
        Hoemann et al. (2020).

        References
        ----------
        Berntson, G., Quigley, K., Jang, J., Boysen, S. (1990). An approach to
        artifact identification: Application to heart period data.
        Psychophysiology, 27(5), 586–598.

        Hegarty-Craver, M. et al. (2018). Automated respiratory sinus
        arrhythmia measurement: Demonstration using executive function
        assessment. Behavioral Research Methods, 50, 1816–1823.

        Hoemann, K. et al. (2020). Context-aware experience sampling reveals
        the scale of variation in affective experience. Scientific
        Reports, 10(1), 1–16.
        """

        def identify_artifacts_hegarty(
            beats_ix: Union[np.ndarray, list], 
            initial_hr: Optional[Union[int, float, Literal['auto']]] = 'auto',
            prev_n: int = 6
        ) -> np.ndarray:
            """Identify locations of artifactual beats in cardiovascular data
            based on the approach by Hegarty-Craver et al. (2018)."""

            ibis = (np.diff(beats_ix) / self.fs) * 1000
            beats = beats_ix[1:]  # drop the first beat
            artifact_beats = []
            valid_beats = [beats_ix[0]]  # assume first beat is valid

            # Set the initial IBI to compare against
            if initial_hr == 'auto':
                successive_diff = np.abs(np.diff(ibis))
                min_diff_ix = np.convolve(
                    successive_diff, np.ones(6) / 6, mode = 'valid').argmin()
                first_ibi = ibis[min_diff_ix:min_diff_ix + 6].mean()
            else:
                first_ibi = 60000 / initial_hr

            for n in range(len(ibis)):
                current_ibi = ibis[n]
                current_beat = beats[n]

                # Check against an estimate of the first N IBIs
                if n < prev_n:
                    if n == 0:
                        ibi_estimate = first_ibi
                    else:
                        next_five = np.insert(ibis[:n], 0, first_ibi)
                        ibi_estimate = np.median(next_five)

                # Check against an estimate of the preceding N IBIs
                else:
                    ibi_estimate = np.median(ibis[n - (prev_n):n])

                # Set the acceptable/valid range of IBIs
                low = (26 / 32) * ibi_estimate
                high = (44 / 32) * ibi_estimate

                if low <= current_ibi <= high:
                    valid_beats.append(current_beat)
                else:
                    artifact_beats.append(current_beat)

            return np.array(valid_beats), np.array(artifact_beats)

        def identify_artifacts_cbd(
            beats_ix: Union[np.ndarray, list], 
            neighbors: int = 5, 
            tol: float = 1
        ) -> np.ndarray:
            """Identify locations of abnormal interbeat intervals (IBIs) using
             the criterion beat difference test by Berntson et al. (1990)."""

            # Derive IBIs from beat indices
            ibis = ((np.ediff1d(beats_ix)) / self.fs) * 1000

            # Compute consecutive absolute differences across IBIs
            ibi_diffs = np.abs(np.ediff1d(ibis))

            # Initialize an array to store "bad" IBIs
            ibi_bad = np.zeros(shape = len(ibis))
            artifact_beats = []

            if len(ibi_diffs) < neighbors:
                neighbors = len(ibis)

            for ii in range(len(ibi_diffs)):

                # If there are not enough neighbors in the beginning
                if ii < int(neighbors / 2) + 1:
                    select = np.concatenate(
                        (ibi_diffs[:ii], ibi_diffs[(ii + 1):(neighbors + 1)]))
                    select_ibi = np.concatenate(
                        (ibis[:ii], ibis[(ii + 1):(neighbors + 1)]))

                # If there are not enough neighbors at the end
                elif (len(ibi_diffs) - ii) < (int(neighbors / 2) + 1) and (
                        len(ibi_diffs) - ii) > 1:
                    select = np.concatenate(
                        (ibi_diffs[-(neighbors - 1):ii], ibi_diffs[ii + 1:]))
                    select_ibi = np.concatenate(
                        (ibis[-(neighbors - 1):ii], ibis[ii + 1:]))

                # If there is only one neighbor left to check against
                elif len(ibi_diffs) - ii == 1:
                    select = ibi_diffs[-(neighbors - 1):-1]
                    select_ibi = ibis[-(neighbors - 1):-1]

                else:
                    select = np.concatenate(
                        (ibi_diffs[ii - int(neighbors / 2):ii],
                         ibi_diffs[(ii + 1):(ii + 1 + int(neighbors / 2))]))
                    select_ibi = np.concatenate(
                        (ibis[ii - int(neighbors / 2):ii],
                         ibis[(ii + 1):(ii + 1 + int(neighbors / 2))]))

                # Calculate the quartile deviation
                QD = self._quartile_deviation(select)

                # Calculate the maximum expected difference (MED)
                MED = 3.32 * QD

                # Calculate the minimal artifact difference (MAD)
                MAD = (np.median(select_ibi) - 2.9 * QD) / 3

                # Calculate the criterion beat difference score
                criterion_beat_diff = (MED + MAD) / 2

                # Find indices of IBIs that fail the CBD check
                if (ibi_diffs[ii]) > tol * criterion_beat_diff:
                    bad_neighbors = int(neighbors * 0.25)
                    end_idx = min(ii + bad_neighbors + 1, len(beats_ix))
                    artifact_beats.extend(beats_ix[ii + 1:end_idx])
                    ibi_bad[ii + 1] = 1

            artifact_beats = np.array(artifact_beats).flatten()
            return artifact_beats

        if method == 'hegarty':
            initial_hr = initial_hr if initial_hr is not None else 'auto'
            prev_n = prev_n if prev_n is not None else 6
            _, artifacts_ix = identify_artifacts_hegarty(
                beats_ix, initial_hr, prev_n)
        elif method == 'cbd':
            neighbors = neighbors if neighbors is not None else 5
            tol = tol if tol is not None else 1
            artifacts_ix = identify_artifacts_cbd(
                beats_ix, neighbors, tol)
        elif method == 'both':
            initial_hr = initial_hr if initial_hr is not None else 'auto'
            prev_n = prev_n if prev_n is not None else 6
            neighbors = neighbors if neighbors is not None else 5
            tol = tol if tol is not None else 1
            _, artifact_hegarty = identify_artifacts_hegarty(
                beats_ix, initial_hr, prev_n)
            artifact_cbd = identify_artifacts_cbd(
                beats_ix, neighbors, tol)
            artifacts_ix = np.union1d(artifact_hegarty, artifact_cbd)
        else:
            raise ValueError(
                'Invalid method. Method must be \'hegarty\', \'cbd\', '
                'or \'both\'.')
        return artifacts_ix

    def get_missing(
        self, data: pd.DataFrame, 
        beats_ix: Union[np.ndarray, list], 
        seg_size: int = 60, 
        min_hr: Union[int, float] = 40,
        ts_col: Optional[str] = None, 
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Summarize the number and proportion of missing beats per segment.

        Parameters
        ----------
        data : pandas.DataFrame
            The DataFrame containing the pre-processed ECG or PPG data.
        beats_ix : array-like
            An array containing the indices of detected beats.
        seg_size : int
            The size of the segment in seconds; by default, 60.
        min_hr : int, float, optional
            The minimum acceptable heart rate against which the number of
            beats in the last partial segment will be compared; by default, 40.
        ts_col : str, optional
            The name of the column containing timestamps; by default, None.
            If a string value is given, the output will contain a timestamps
            column.
        show_progress : bool, optional
            Whether to display a progress by while the function runs; by
            default, True.

        Returns
        -------
        missing : pandas.DataFrame
            A DataFrame with detected, expected, and missing numbers of
            beats per segment.
        """
        seconds = self.get_seconds(data, beats_ix, ts_col, show_progress)
        seconds.index = seconds.index.astype(int)

        n_seg = ceil(len(seconds) / seg_size)
        segments = pd.Series(np.arange(1, n_seg + 1))
        # n_expected = (
        #         seconds.groupby(seconds.index // seg_size)['Mean HR'].median() * (seg_size / 60)
        # ).fillna(0).astype(int)
        raw_expected = seconds['Mean HR'].rolling(
            window = seg_size + 1, min_periods = 1, center = True).median()
        n_expected = raw_expected.groupby(raw_expected.index // seg_size).median() * (seg_size / 60)
        n_detected = seconds.groupby(
            seconds.index // seg_size)['N Beats'].sum()
        n_missing = (n_expected - n_detected).clip(lower = 0)
        perc_missing = (n_missing / n_expected) * 100

        # Handle last partial segment of data
        last_seg_len = len(seconds) % seg_size
        if last_seg_len > 0:
            last_detected = n_detected.iloc[-1]
            last_expected_ratio = min_hr / n_expected.iloc[:-1].median()
            last_expected = last_expected_ratio * last_seg_len
            if last_expected > last_detected:
                last_n_missing = last_expected - last_detected
                last_perc_missing = (last_n_missing / last_expected) * 100
            else:
                last_perc_missing = 0
                last_n_missing = 0
            n_expected.iloc[-1] = last_expected
            n_missing.iloc[-1] = last_n_missing
            perc_missing.iloc[-1] = last_perc_missing

        if ts_col is not None:
            timestamps = seconds.groupby(
                seconds.index // seg_size).first()['Timestamp']
            missing = pd.concat([
                segments,
                timestamps,
                n_detected.astype(int),
                n_expected.astype(int),
                n_missing.astype(int),
                round(perc_missing, 2),
            ], axis = 1)
            missing.columns = [
                'Segment',
                'Timestamp',
                'N Detected',
                'N Expected',
                'N Missing',
                '% Missing',
            ]
        else:
            missing = pd.concat([
                segments,
                n_detected.astype(int),
                n_expected.astype(int),
                n_missing.astype(int),
                round(perc_missing, 2),
            ], axis = 1)
            missing.columns = [
                'Segment',
                'N Detected',
                'N Expected',
                'N Missing',
                '% Missing',
            ]
        return missing

    def correct_interval(
        self, 
        beats_ix: Union[np.ndarray, list], 
        initial_hr: Union[int, float, Literal['auto']] = 'auto', 
        prev_n: int = 6, 
        min_bpm: int = 40, 
        max_bpm: int = 200, 
        hr_estimate_window: int = 6
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
        Correct artifactual beats in cardiovascular data based
        on the approach by Hegarty-Craver et al. (2018).

        Parameters
        ----------
        beats_ix : array_like
            An array containing the indices of detected beats.
        initial_hr : int, float, 'auto', optional
            The heart rate value for the first interbeat interval (IBI) to be
            validated against; by default, automatically set ('auto').
        prev_n : int, optional
            The number of preceding IBIs to validate against; by default, 6.
        min_bpm : int, optional
            The minimum possible heart rate in beats per minute (bpm); by default, 40.
        max_bpm : int, optional
            The maximum possible heart rate in beats per minute (bpm); by default, 200.
        hr_estimate_window : int, optional
            The window size for estimating the heart rate if initial_hr = 'auto'; by default, 6.

        Returns
        -------
        original: pandas.DataFrame
            A data frame containing the original IBIs (millisecond-based and index-based) and beat indices.
        corrected: pandas.DataFrame
            A data frame containing the corrected IBIs (millisecond-based and index-based) and beat indices.

        References
        ----------
        Hegarty-Craver, M. et al. (2018). Automated respiratory sinus
        arrhythmia measurement: Demonstration using executive function
        assessment. Behavioral Research Methods, 50, 1816–1823.
        '''
        global MIN_BPM, MAX_BPM
        MIN_BPM = min_bpm
        MAX_BPM = max_bpm
        MAX_COUNT = 10              # maximum number of consecutive corrections

        DEBUGGING = False
        
        ibis = np.diff(beats_ix)
        beats = beats_ix[1:]        # drop the first beat

        short_threshold = 24 / 32
        long_threshold = 44 / 32
        extra_threshold = 52 / 32

        global cnt, corrected_ibis, corrected_beats, corrected_flags
        global prev_ibi, prev_beat, prev_flag, current_ibi, current_beat, current_flag, correction_flags
        cnt = 0                     # increment when correcting the ibi and decrement when accepting the ibi
        prev_ibi = 0
        prev_beat = 0
        prev_flag = None
        current_ibi = 0
        current_beat = 0
        current_flag = None
        corrected_ibis = []
        corrected_beats = []
        corrected_flags = []
        correction_flags = [0 for i in range(len(beats))]

        ## DELETE THIS LATER
        global n_insert
        n_insert = 0

        class MaxNFifo:
            '''
            A class for FIFO with N elements at maximum.

            Parameters/Attributes
            ---------------------
            prev_n : int
                The maximum number of elements in the FIFO.
            '''
            def __init__(self, prev_n: int, item: Optional[int] = None):
                '''
                Initialize the FIFO object.
                
                Parameters
                ---------------------
                prev_n : int
                    The maximum number of elements in the FIFO.
                item : int, optional
                    The initial item to add to the FIFO; by default, None.
                    The item is added twice if it is not None.
                '''
                self.prev_n = prev_n
                if item is not None:
                    self.queue = [item, item]
                else:
                    self.queue = []

            def push(self, item: int):
                '''
                Push an item to the FIFO. If the number of elements exceeds the 
                maximum, remove the first element.

                Parameters
                ---------------------
                item : int
                    The item to add to the FIFO.
                '''
                self.queue.append(item)
                if len(self.queue) > self.prev_n + 1:
                    self.queue.pop(0)

            def get_queue(self):
                '''
                Return the FIFO queue.

                Return
                ---------------------
                queue : list
                '''
                return self.queue
            
            def change_last(self, item: int):
                '''
                Change the last item in the FIFO queue.

                Parameters
                ---------------------
                item : int
                    The new item to replace the last item in the queue.
                '''
                self.queue[-1] = item
            
            def reset(self, item: Optional[int] = None):
                '''
                Reset the FIFO queue. 
                If an item is given, reset the queue with the item. If not, 
                reset the queue with an empty list.

                Parameters
                ---------------------
                item: int, optional
                    The item to add to the FIFO; by default, None.
                    The item is added twice if it is not None.
                '''
                if item is None:
                    self.queue = []
                else:
                    self.queue = [item, item]
        
        # Set the initial IBI to compare against
        global prev_ibis_fifo, first_ibi, correction_failed
        if initial_hr == 'auto':
            successive_diff = np.abs(np.diff(ibis))
            min_diff_ix = np.convolve(successive_diff, np.ones(hr_estimate_window) / 
                                      hr_estimate_window, mode = 'valid').argmin()
            first_ibi = ibis[min_diff_ix:min_diff_ix + hr_estimate_window].mean()
        else:
            first_ibi = self.fs * 60 / initial_hr
        
        prev_ibis_fifo = MaxNFifo(prev_n, first_ibi)  # FIFO for the previous n+1 IBIs
        correction_failed = MaxNFifo(prev_n - 1)      # Store whether the correction failed for the last n IBIs

        def estimate_ibi(prev_ibis: np.ndarray):
            '''
            Estimate IBI based on the previous IBIs.
            
            Parameters
            ---------------------
            prev_ibis: array-like
                A list of prev_n number of previous IBIs.
            
            Returns
            ---------------------
            estimated_ibi : int
            '''
            return np.median(prev_ibis)

        def return_flag(current_ibi: int, prev_ibis: Optional[np.ndarray] = None):
            '''
            Return whether the current IBI is correct, short, long, or extra 
            long based on the previous IBIs.
                Correct: 26/32 - 44/32 of the estimated IBI
                Short: < 26/32 of the estimated IBI
                Long: > 44/32 and < 54/32 of the estimated IBI
                Extra Long: > 54/32 of the estimated IBI
            
            Parameters
            ---------------------
            current_ibi: int
                current IBI value in the number of indices.
            prev_ibis: array-like, optional
                A list of prev_n number of previous IBIs.
            
            Returns
            ---------------------
            flag : str
                The flag of the current IBI: 'Correct', 'Short', 'Long', or 
                'Extra Long'.
            '''
            # Calculate the estimated IBI
            estimated_ibi = estimate_ibi(prev_ibis)

            # Set the acceptable/valid range of IBIs
            low = short_threshold * estimated_ibi
            high = long_threshold * estimated_ibi
            extra = extra_threshold * estimated_ibi

            # flag the ibi: correct, short, long, or extra long
            if low <= current_ibi <= high:
                flag = 'Correct'
            elif current_ibi < low:
                flag = 'Short'
            elif current_ibi > high and current_ibi < extra:
                flag = 'Long'
            else:
                flag = 'Extra Long'
            
            # print('current:', current_ibi, ' prev:', prev_ibis, ' flag:', flag)
            
            return flag
        
        def acceptance_check(corrected_ibi: int, prev_ibis: np.ndarray):
            '''
            Check if the corrected IBI is acceptable (falls within the 
            27/32 – 42/32 of the estimated IBI).

            Parameters
            ---------------------
            corrected_ibi: int
                The corrected IBI value.
            prev_ibis: array_like
                A list of prev_n number of previous IBIs.
            
            Returns
            ---------------------
            bool
                True if the corrected IBI is within the acceptable range, False 
                otherwise.
            '''
            # Calculate the estimated IBI
            estimated_ibi = estimate_ibi(prev_ibis)
            
            # Set the acceptable/valid range of IBIs
            low = short_threshold * estimated_ibi
            high = long_threshold * estimated_ibi

            # If the corrected value is within the range, return True
            if corrected_ibi >= low and corrected_ibi <= high:
            #if corrected_ibi <= high:
                return True
            else:
                return False

        def accept_ibi(n: int, correction_failed_flag: int = 0):
            '''
            Accept the current IBI without correction.

            Parameters
            ---------------------
            n : int
                The index of the current IBI.
            correction_failed_flag : int, optional
                Flag to indicate whether the correction failed for the 
                current IBI; by default, 0. If the flag is 1, the correction 
                failed for the current IBI.
            '''
            global prev_ibis_fifo, cnt, correction_failed
            global corrected_ibis, corrected_beats, corrected_flags
            global prev_ibi, prev_beat, prev_flag, current_ibi, current_beat, current_flag
            
            # Check if the previous IBI is within the limits before accepting the current IBI
            check_limits(n)

            # Fix the previous IBI
            corrected_ibis.append(prev_ibi)
            corrected_beats.append(prev_beat)
            corrected_flags.append(prev_flag)

            # Add the previous IBI to the queue
            prev_ibis_fifo.push(prev_ibi)

            # Update the previous IBI to the current IBI
            prev_ibi = current_ibi
            prev_beat = current_beat
            prev_flag = current_flag
            
            # Decrement the counter
            cnt = max(0, cnt-1)
            if DEBUGGING:
                print('accepted:', current_ibi, ' flag:', current_flag, ' based on ', prev_ibis_fifo.get_queue()[1:])
            # If the correction failed for the current IBI, push 1 to the correction_failed FIFO, otherwise push 0
            if correction_failed_flag == 0:
                correction_failed.push(0)
            else:
                correction_failed.push(1)
        
        def add_prev_and_current(n: int):
            '''
            Add the previous and current IBIs if the sum is less than 42/32 of the estimated IBI.

            Parameters
            ---------------------
            n : int
                The index of the current IBI.
            '''
            global prev_ibis_fifo, cnt
            global corrected_ibis, corrected_beats, corrected_flags
            global prev_ibi, prev_beat, prev_flag, current_ibi, current_beat, current_flag, correction_flags
            
            # Add the previous and current IBIs
            corrected_ibi = prev_ibi + current_ibi

            # Check if the corrected IBI is acceptable
            if acceptance_check(corrected_ibi, prev_ibis_fifo.get_queue()[1:]):
                # Update the current IBI to the corrected IBI
                current_ibi = corrected_ibi
                current_beat = current_beat
                current_flag = return_flag(current_ibi, prev_ibis_fifo.get_queue()[1:])
                
                if n == 1:
                    # Update the previous IBI to the current IBI
                    prev_ibi = current_ibi
                    prev_beat = current_beat
                    prev_flag = current_flag

                else:
                    # Pull up the second previous IBI as previous IBI
                    prev_ibi = corrected_ibis[-1]
                    prev_beat = corrected_beats[-1]
                    prev_flag = corrected_flags[-1]
                    
                    # Check if the previous IBI is within the limits before accepting the current IBI
                    check_limits(n)
                    
                    # Check_limits function may update the previous IBI pulled, so update the value
                    corrected_ibis[-1] = prev_ibi
                    corrected_beats[-1] = prev_beat
                    corrected_flags[-1] = prev_flag

                    # Update the last IBI value in the queue
                    prev_ibis_fifo.change_last(prev_ibi)

                    # Update the previous IBI to the current IBI
                    prev_ibi = current_ibi
                    prev_beat = current_beat
                    prev_flag = current_flag

                # Flag that previous and current IBIs are corrected
                correction_flags[n-1] = 1
                correction_flags[n] = 1

                # Increment the counter
                cnt += 1

                if DEBUGGING:
                    print('added:', current_ibi, ' flag:', current_flag, ' based on ', prev_ibis_fifo.get_queue()[1:])
            else:
                if DEBUGGING:
                    print('acceptance check failed for adding: ', corrected_ibi)
                # If the corrected IBI is not acceptable, accept the current IBI
                accept_ibi(n, correction_failed_flag=1)
        
        def add_secondprev_and_prev(n: int):
            '''
            Add the second previous and previous IBIs if the sum is less than 42/32 of the estimated IBI.

            Parameters
            ---------------------
            n : int
                The index of the current IBI.
            '''
            global prev_ibis_fifo, cnt
            global corrected_ibis, corrected_beats, corrected_flags
            global prev_ibi, prev_beat, prev_flag, current_ibi, current_beat, current_flag, correction_flags
            
            # Add the previous and current IBIs
            corrected_ibi = corrected_ibis[-1] + prev_ibi

            # Check if the corrected IBI is acceptable
            # Use IBIs before the second previous IBI
            if acceptance_check(corrected_ibi, prev_ibis_fifo.get_queue()[:-2]):
                # Update the current IBI to the corrected IBI
                
                # Pull up the second previous IBI as previous IBI
                prev_ibi = corrected_ibi
                prev_beat = prev_beat
                prev_flag = return_flag(prev_ibi, prev_ibis_fifo.get_queue()[:-2])
                
                # Check if the previous IBI is within the limits before accepting the current IBI
                check_limits(n)
                
                # Update the value
                corrected_ibis[-1] = prev_ibi
                corrected_beats[-1] = prev_beat
                corrected_flags[-1] = prev_flag

                # Update the last IBI value in the queue
                prev_ibis_fifo.change_last(prev_ibi)

                # Update the previous IBI to the current IBI
                prev_ibi = current_ibi
                prev_beat = current_beat
                prev_flag = current_flag

                # Flag that previous and current IBIs are corrected
                correction_flags[n-2] = 1
                correction_flags[n-1] = 1

                # Increment the counter
                cnt += 1

                if DEBUGGING:
                    print('added second prev + prev:', prev_ibi, ' flag:', prev_flag, ' based on ', prev_ibis_fifo.get_queue()[:-2])
            else:
                if DEBUGGING:
                    print('acceptance check failed for adding second prev + prev: ', corrected_ibi)
                # If the corrected IBI is not acceptable, accept the current IBI
                accept_ibi(n, correction_failed_flag=1)
                
        def insert_interval(n: int):
            '''
            Split the (previous IBI + current IBI) into multiple intervals. 
            The number of splits is determined based on the initial_hr parameter.
            
            Parameters
            ---------------------
            n : int
                The index of the current IBI.
            '''
            global prev_ibis_fifo, cnt, first_ibi
            global corrected_ibis, corrected_beats, corrected_flags
            global prev_ibi, prev_beat, prev_flag, current_ibi, current_beat, current_flag, correction_flags

            ## DELETE THIS LATER
            global n_insert
            
            # Calculate the number of splits
            n_split = round((prev_ibi + current_ibi) / estimate_ibi(prev_ibis_fifo.get_queue()[1:]), 0).astype(int)

            # Calculate the new IBI
            ibi = floor((prev_ibi + current_ibi) / n_split)

            # Check if the corrected IBI is acceptable
            if acceptance_check(ibi, prev_ibis_fifo.get_queue()[1:]):
                # Fix inserted IBIs other than previous/current IBIs
                for i in range(n_split - 2):
                    corrected_ibis.append(ibi)
                    corrected_flags.append(return_flag(ibi, prev_ibis_fifo.get_queue()[1:]))
                    if n == 1 and i == 0:
                        corrected_beats.append(beats_ix[0] + ibi)
                    else:
                        corrected_beats.append(corrected_beats[-1] + ibi)
                    # Add to the queue
                    prev_ibis_fifo.push(ibi)

                # Update the previous IBI
                prev_ibi = ibi
                prev_beat = corrected_beats[-1] + ibi
                prev_flag = return_flag(ibi, prev_ibis_fifo.get_queue()[:-1])

                # Update the current IBI
                current_ibi = current_beat - prev_beat
                current_flag = return_flag(ibi, prev_ibis_fifo.get_queue()[1:])

                # Check if the previous IBI is within the limits
                check_limits(n)

                # Fix the previous IBI
                corrected_ibis.append(prev_ibi)
                corrected_beats.append(prev_beat)
                corrected_flags.append(prev_flag)

                # Add to the queue
                prev_ibis_fifo.push(prev_ibi)
                
                # Update the previous IBI to the current IBI
                prev_ibi = current_ibi
                prev_beat = current_beat
                prev_flag = current_flag

                # Flag that previous and current IBIs are corrected
                correction_flags[n-1] = 1
                correction_flags[n] = 1

                # Increment the counter by n_split - 1 in this case
                cnt += n_split - 1

                ## DELETE THIS LATER
                n_insert += n_split - 2

                if DEBUGGING:
                    print('inserted ',n_split - 2, ' intervals: ', ibi, ' flag:', current_flag, ' based on ', prev_ibis_fifo.get_queue()[1:])
            else:
                if DEBUGGING:
                    print('acceptance check failed for inserting: ', ibi)
                # If the corrected IBI is not acceptable, accept the current IBI
                accept_ibi(n, correction_failed_flag=1)

        def average_prev_and_current(n: int):
            '''
            Average the previous and current IBIs.

            Parameters
            ---------------------
            n : int
                The index of the current IBI.
            '''
            global prev_ibis_fifo, cnt
            global corrected_ibis, corrected_beats, corrected_flags
            global prev_ibi, prev_beat, prev_flag, current_ibi, current_beat, current_flag, correction_flags
            
            # Average the previous and current IBIs
            ibi = floor((prev_ibi + current_ibi) / 2)
            
            # Check if the corrected IBI is acceptable
            if acceptance_check(ibi, prev_ibis_fifo.get_queue()[1:]):
                # Update the previous and current IBI
                prev_ibi = ibi
                if n == 1:
                    prev_beat = beats_ix[0] + ibi
                else:
                    prev_beat = corrected_beats[-1] + ibi
                prev_flag = return_flag(ibi, prev_ibis_fifo.get_queue()[:-1])
                current_ibi = current_beat - prev_beat
                current_flag = return_flag(ibi, prev_ibis_fifo.get_queue()[1:])

                # Check if the previous IBI is within the limits
                check_limits(n)

                # Fix the previous IBI
                corrected_ibis.append(prev_ibi)
                corrected_beats.append(prev_beat)
                corrected_flags.append(prev_flag)

                # Add to the queue
                prev_ibis_fifo.push(prev_ibi)

                # Update the previous IBI to the current IBI
                prev_ibi = current_ibi
                prev_beat = current_beat
                prev_flag = current_flag
                
                # Flag that previous and current IBIs are corrected
                correction_flags[n-1] = 1
                correction_flags[n] = 1

                # Increment the counter
                cnt += 1

                if DEBUGGING:
                    print('averaged:', ibi, ' flag:', current_flag, ' based on ', prev_ibis_fifo.get_queue()[1:])
            else:
                if DEBUGGING:
                    print('acceptance check failed for averaging: ', ibi)
                accept_ibi(n, correction_failed_flag=1)

        def check_limits(n: int):
            '''
            Check if the previous IBI (n-1) is within the limits.
            If it is longer the maximum IBI, shorten the previous IBI and lengthen the current IBI.
            If it is shorter than the minimum IBI, lengthen the previous IBI and shorten the current IBI.

            Parameters
            ---------------------
            n : int
                The index of the current IBI.
            '''
            global prev_ibis_fifo, cnt
            global corrected_ibis, corrected_beats, corrected_flags
            global prev_ibi, prev_beat, prev_flag, current_ibi, current_beat, current_flag, correction_flags
            MIN_IBI = floor(self.fs * 60 / MAX_BPM)         # minimum IBI in indices
            MAX_IBI = floor(self.fs * 60 / MIN_BPM)         # maximum IBI in indices

            # If the previous IBI is shorter than the minimum IBI, lengthen the previous IBI and shorten the current IBI
            if prev_ibi < MIN_IBI:
                remainder = MIN_IBI - prev_ibi
                prev_beat = prev_beat + remainder
                prev_ibi = MIN_IBI
                prev_flag = return_flag(prev_ibi, prev_ibis_fifo.get_queue()[:-1])
                current_ibi = current_ibi - remainder
                current_flag = return_flag(current_ibi, prev_ibis_fifo.get_queue()[1:])

                # Flag that previous and current IBIs are corrected
                correction_flags[n-1] = 1
                correction_flags[n] = 1

                # Increment the counter
                cnt += 1

                if DEBUGGING:
                    print('Shorter than the minimum IBI and corrected: ', prev_ibi, ' ', prev_flag, ' | ', current_ibi, ' ', current_flag)

            # If the previous IBI is longer than the maximum IBI, shorten the previous IBI and lengthen the current IBI
            elif prev_ibi > MAX_IBI:
                remainder = prev_ibi - MAX_IBI
                prev_beat = prev_beat - remainder
                prev_ibi = MAX_IBI
                prev_flag = return_flag(prev_ibi, prev_ibis_fifo.get_queue()[:-1])
                current_ibi = current_ibi + remainder
                current_flag = return_flag(current_ibi, prev_ibis_fifo.get_queue()[1:])

                # Flag that previous and current IBIs are corrected
                correction_flags[n-1] = 1
                correction_flags[n] = 1

                # Increment the counter
                cnt += 1

                if DEBUGGING:
                    print('Longer than the maximum IBI and corrected: ', prev_ibi, ' ', prev_flag, ' | ', current_ibi, ' ', current_flag)
            return

        for n in range(len(ibis)):
            current_ibi = ibis[n]
            current_beat = beats[n]
            # If the adaptive threshold is enabled, calculate the thresholds
                
            # Accept the first ibi
            if n == 0:
                current_flag = return_flag(current_ibi, prev_ibis = prev_ibis_fifo.get_queue())
                # Update the previous IBI to the current IBI
                prev_ibi = current_ibi
                prev_beat = current_beat
                prev_flag = current_flag
            
            # If the counter reaches the maximum count, accept the current IBI
            elif cnt > MAX_COUNT:
                current_flag = return_flag(current_ibi, prev_ibis = prev_ibis_fifo.get_queue()[:-1])
                accept_ibi(n)
                # decrement by 2 when cnt > MAX_COUNT
                cnt -= 1
            
            else:
                current_flag = return_flag(current_ibi, prev_ibis = prev_ibis_fifo.get_queue()[:-1])
                # If the current ibi is correct
                if DEBUGGING:
                    print('n:', n)
                    print('prev:', prev_ibi, ' ', prev_flag, ' | current:', current_ibi, ' ', current_flag)
                if current_flag == 'Correct':
                    if prev_flag == 'Correct' or prev_flag == 'Long':
                        accept_ibi(n)                           # If the previous ibi is correct or long, then accept the current ibi
                    elif prev_flag == 'Short':
                        if n == 1:
                            add_prev_and_current(n) 
                        else:
                            if corrected_ibis[-1] > current_ibi:
                                add_prev_and_current(n)                 # If the previous ibi is short, add previous and current intervals
                            else:
                                add_secondprev_and_prev(n)
                    elif prev_flag == 'Extra Long':
                        insert_interval(n)                      # If the previous ibi is extra long, split the previous and current intervals
                # If the current ibi is short
                elif current_flag == 'Short':
                    if prev_flag == 'Correct':
                        accept_ibi(n)                           # If the previous ibi is correct, accept previous
                    elif prev_flag == 'Short':
                        add_prev_and_current(n)                 # If the previous ibi is short, add previous and current intervals
                    elif prev_flag == 'Long' or prev_flag == 'Extra Long':
                        average_prev_and_current(n)             # If the previous ibi is long or extra long, average the previous and current intervals
                # If the current ibi is long
                elif current_flag == 'Long':
                    if prev_flag == 'Correct' or prev_flag == 'Long':
                        accept_ibi(n)                           # If the previous ibi is correct or long, accept previous
                    elif prev_flag == 'Short':
                        average_prev_and_current(n)             # If the previous ibi is short, average previous and current intervals
                    elif prev_flag == 'Extra Long':
                        insert_interval(n)                      # If the previous ibi is extra long, split the previous and current intervals
                # If the current ibi is extra long
                elif current_flag == 'Extra Long':
                    if prev_flag == 'Correct' or prev_flag == 'Long' or prev_flag == 'Extra Long':
                        insert_interval(n)                      # If the previous ibi is correct, long, or extra long, split the previous and current intervals
                    elif prev_flag == 'Short':
                        average_prev_and_current(n)             # If the previous ibi is short, average previous and current intervals

            # If more than 3 corrections are made in the last prev_n IBIs, reset the FIFO
            if sum(correction_failed.get_queue()) >= 3:
                prev_ibis_fifo.reset(first_ibi)

        # Add the last beat
        corrected_ibis.append(current_ibi)
        corrected_beats.append(current_beat)
        corrected_flags.append(current_flag)

        correction_flags = np.array(correction_flags).astype(int)

        # Convert the IBIs to milliseconds
        original_ibis_ms = np.round((np.array(ibis) / self.fs) * 1000, 2)
        
        # Add the first beat and create a dataframe
        original = pd.DataFrame(
            {'Original IBI (ms)': np.insert(original_ibis_ms, 0, np.nan),
            'Original IBI (index)': np.insert(ibis.astype(object), 0, np.nan),
            'Original Beat': np.insert(beats, 0, beats_ix[0]),
            'Correction': np.insert(correction_flags, 0, 0)}
        )
        
        corrected_ibis_ms = np.round((np.array(corrected_ibis) / self.fs) * 1000, 2)

        corrected_ibis = np.array(corrected_ibis).astype(object)
        corrected_flags = np.array(corrected_flags).astype(object)

        # Add the first beat and create a dataframe
        corrected = pd.DataFrame(
            {'Corrected IBI (ms)': np.insert(corrected_ibis_ms, 0, np.nan),
            'Corrected IBI (index)': np.insert(corrected_ibis, 0, np.nan), 
            'Corrected Beat': np.insert(corrected_beats, 0, beats_ix[0]),
            'Flag': np.insert(corrected_flags, 0, np.nan)}
        )

        return original, corrected
    
    def get_corrected(
        self, 
        beats_ix: Union[np.ndarray, list], 
        seg_size: int, 
        initial_hr: Union[int, float, Literal['auto']] = 'auto', 
        prev_n: int = 6, 
        min_bpm: int = 40, 
        max_bpm: int = 200, 
        hr_estimate_window: int = 6
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get the corrected interbeat intervals (IBIs) and beat indices.
    
        Parameters
        ----------
        beats_ix : array_like
            An array containing the indices of detected beats.
        seg_size : int, optional
            The size of the segment in seconds; by default, 60.
        initial_hr : int, float, optional
            The heart rate value for the first interbeat interval (IBI) to be
            validated against; by default, 80 bpm (750 ms).
        prev_n : int, optional
            The number of preceding IBIs to validate against; by default, 6.
        min_bpm : int, optional
            The minimum possible heart rate in beats per minute (bpm); by default, 40.
        max_bpm : int, optional
            The maximum possible heart rate in beats per minute (bpm); by default, 200.
        hr_estimate_window : int, optional
            The window size for estimating the heart rate if initial_hr = 'auto'; by default, 6.
    
        Returns
        -------
        original : pd.DataFrame
            A DataFrame containing the original IBIs, their indices, and any applied corrections.
        corrected : pd.DataFrame
            A DataFrame containing the corrected IBIs, their indices, and correction flags.
        summary : pd.DataFrame
            A DataFrame summarizing the number and percentage of corrected beats per segment, 
            as well as the count of different types of corrections (Correct, Short, Long, Extra Long).
        """
        # Get the corrected IBIs and beat indices
        original, corrected = self.correct_interval(
            beats_ix = beats_ix, 
            initial_hr = initial_hr, 
            prev_n = prev_n, 
            min_bpm = min_bpm, 
            max_bpm = max_bpm, 
            hr_estimate_window = hr_estimate_window)
    
        # Get the segment number for each beat
        for row in original.iterrows():
            seg = ceil(row[1].loc['Original Beat'] / (seg_size * self.fs))
            original.loc[row[0], 'Segment'] = seg
        for row in corrected.iterrows():
            seg = ceil(row[1].loc['Corrected Beat'] / (seg_size * self.fs))
            corrected.loc[row[0], 'Segment'] = seg
        original['Segment'] = original['Segment'].astype(pd.Int64Dtype())
        corrected['Segment'] = corrected['Segment'].astype(pd.Int64Dtype())
        
        # Get the number and percentage of corrected beats in each segment
        original_seg = original.groupby('Segment')['Correction'].sum().astype(pd.Int64Dtype())
        original_seg = pd.DataFrame(original_seg.reset_index(name = '# Corrected'))
        original_seg_nbeats = original.groupby('Segment')['Correction'].count().astype(pd.Int64Dtype())
        original_seg_nbeats = pd.DataFrame(original_seg_nbeats.reset_index(name = '# Beats'))
        original_seg = original_seg.merge(original_seg_nbeats, on = 'Segment')
        original_seg['% Corrected'] = round((original_seg['# Corrected'] / original_seg['# Beats']) * 100, 2)
        original_seg.drop('# Beats', axis = 1, inplace = True)
    
        # Get the number of each flag (Correct/Short/Long/Extra Long) in each segment
        corrected_seg = corrected.groupby('Segment')['Flag'].value_counts().astype(pd.Int64Dtype())
        corrected_seg = pd.DataFrame(corrected_seg.reset_index(name = 'Count'))
        corrected_seg = corrected_seg.pivot(index = 'Segment', columns = 'Flag', values = 'Count').reset_index().fillna(0)
        corrected_seg.columns.name = None
        corrected_seg = corrected_seg.rename_axis(None, axis = 1)
    
        combined = pd.merge(corrected_seg, original_seg, on = 'Segment')
    
        return original, corrected, combined

    def get_seconds(
        self, 
        data: pd.DataFrame, 
        beats_ix: Union[np.ndarray, list], 
        ts_col: Optional[str] = None, 
        show_progress: bool = True
    ) -> pd.DataFrame:
        """Get second-by-second HR, IBI, and beat counts from ECG or PPG data
        according to the approach by Graham (1978).

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the pre-processed ECG or PPG data.
        beats_ix : array-like
            An array containing the indices of detected beats.
        ts_col : str, optional
            The name of the column containing timestamps; by default, None.
            If a string value is given, the output will contain a timestamps
            column.
        show_progress : bool, optional
            Whether to display a progress bar while the function runs; by
            default, True.

        Returns
        -------
        interval_data : pd.DataFrame
            A DataFrame containing second-by-second HR and IBI values.

        Notes
        -----
        Rows with `NaN` values in the resulting DataFrame `interval_data`
        denote seconds during which no beats in the data were detected.

        References
        ----------
        Graham, F. K. (1978). Constraints on measuring heart rate and period
        sequentially through real and cardiac time. Psychophysiology, 15(5),
        492–495.
        """

        df = data.copy()
        temp_beat = '_temp_beat'
        df.index = df.index.astype(int)
        df.loc[beats_ix, temp_beat] = 1

        interval_data = []

        # Iterate over each second
        s = 1
        for i in tqdm(range(0, len(df), self.fs), disable = not show_progress):

            # Get data at the current second and evaluation window
            current_sec = df.iloc[i:(i + self.fs)]
            if i == 0:
                # Look at current and next second
                window = df.iloc[:(i + self.fs)]
            else:
                # Look at previous, current, and next second
                window = df.iloc[(i - self.fs):(min(i + self.fs, len(df)))]

            # Get mean IBI and HR values from the detected beats
            current_beats = current_sec[current_sec[temp_beat] == 1].index.values
            window_beats = window[window[temp_beat] == 1].index.values
            ibis = np.diff(window_beats) / self.fs * 1000
            if len(ibis) == 0:
                mean_ibi = np.nan
                mean_hr = np.nan
            else:
                mean_ibi = np.mean(ibis)
                hrs = 60000 / ibis
                r_hrs = 1 / hrs
                mean_hr = 1 / np.mean(r_hrs)

            # Append values for the current second
            if ts_col is not None:
                interval_data.append({
                    'Second': s,
                    'Timestamp': current_sec.iloc[0][ts_col],
                    'Mean HR': mean_hr,
                    'Mean IBI': mean_ibi,
                    'N Beats': len(current_beats)
                })
            else:
                interval_data.append({
                    'Second': s,
                    'Mean HR': mean_hr,
                    'Mean IBI': mean_ibi,
                    'N Beats': len(current_beats)
                })

            s += 1
        interval_data = pd.DataFrame(interval_data)
        return interval_data

    def plot_missing(
        self, 
        df: pd.DataFrame, 
        invalid_thresh: Union[int, float] = 30, 
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Plot detected and missing beat counts.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing SQA metrics per segment.
        invalid_thresh : int, float
            The minimum number of beats detected for a segment to be 
            considered valid; by default, 30.
        title : str, optional

        Returns
        -------
        fig : plotly.graph_objects.Figure
            A Plotly bar chart showing the number of detected and missing 
            beats per segment. Invalid segments are annotated with a red 
            exclamation mark.
        """
        max_beats = ceil(df['N Detected'].max() / 10) * 10
        nearest = ceil(max_beats / 2) * 2
        dtick_value = nearest / 5

        fig = go.Figure(
            data = [
                go.Bar(
                    x = df['Segment'],
                    y = df['N Expected'],
                    name = 'Missing',
                    marker = dict(color = '#f2816d'),
                    hovertemplate = '<b>Segment %{x}:</b> %{customdata:.0f} '
                                    'missing<extra></extra>'),
                go.Bar(
                    x = df['Segment'],
                    y = df['N Detected'],
                    name = 'Detected',
                    marker = dict(color = '#313c42'),
                    hovertemplate = '<b>Segment %{x}:</b> %{y:.0f} '
                                    'detected<extra></extra>')
            ]
        )
        fig.data[0].update(customdata = df['N Missing'])

        # Annotate invalid segments
        invalid = []
        for i, (segment_num, n_detected) in enumerate(
                zip(df['Segment'], df['N Detected'])):
            if n_detected < invalid_thresh:
                annotation = dict(
                    x = segment_num,
                    y = n_detected,
                    text = '<b>!</b>',
                    showarrow = False,
                    font = dict(color = '#db0f0f', size = 20),
                    yshift = 10,
                )
                invalid.append(annotation)
        if invalid:
            fig.add_annotation(
                text = '<span style="color: #db0f0f"><b>!</b></span>  '
                       'Invalid Number of Beats ',
                align = 'right',
                showarrow = False,
                xref = 'paper',
                yref = 'paper',
                x = 1,
                y = 1.3)

        fig.update_layout(
            xaxis_title = 'Segment Number',
            xaxis = dict(tickmode = 'linear', dtick = 1),
            yaxis = dict(
                title = 'Number of Beats',
                range = [0, max_beats],
                dtick = dtick_value),
            legend = dict(
                orientation = 'h',
                yanchor = 'bottom',
                y = 1.0,
                xanchor = 'right',
                x = 1.0),
            annotations = invalid,
            font = dict(family = 'Poppins', size = 13),
            height = 289,
            margin = dict(t = 70, r = 20, l = 40, b = 65),
            barmode = 'overlay',
            template = 'simple_white',
        )
        if title is not None:
            fig.update_layout(
                title = title
            )
        return fig

    def plot_artifact(
        self, df: pd.DataFrame, 
        invalid_thresh: Union[int, float] = 30, 
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Plot detected and artifact beat counts.

        Parameters
        ----------
        df : pandas.DataFrame()
            The DataFrame containing SQA metrics per segment.
        invalid_thresh : int, float, optional
            The minimum number of beats detected for a segment to be 
            considered valid; by default, 30.
        title : str, optional

        Returns
        -------
        fig : plotly.graph_objects.Figure
            A Plotly bar chart showing the number of artifactual beats per 
            segment. Invalid segments are annotated with a red exclamation 
            mark.
        
        """
        max_beats = ceil(df['N Detected'].max() / 10) * 10
        nearest = ceil(max_beats / 2) * 2
        dtick_value = nearest / 5

        fig = go.Figure(
            data = [
                go.Bar(
                    x = df['Segment'],
                    y = df['N Detected'],
                    name = 'Detected',
                    marker = dict(color = '#313c42'),
                    hovertemplate = '<b>Segment %{x}:</b> %{y:.0f} '
                                    'detected<extra></extra>'),
                go.Bar(
                    x = df['Segment'],
                    y = df['N Artifact'],
                    name = 'Artifact',
                    marker = dict(color = '#f2b463'),
                    hovertemplate = '<b>Segment %{x}:</b> %{y:.0f} '
                                    'artifact<extra></extra>')
            ],
        )

        # Annotate invalid segments
        invalid = []
        for i, (segment_num, n_detected) in enumerate(
                zip(df['Segment'], df['N Detected'])):
            if n_detected < invalid_thresh:
                annotation = dict(
                    x = segment_num,
                    y = n_detected,
                    text = '<b>!</b>',
                    showarrow = False,
                    font = dict(color = '#db0f0f', size = 20),
                    yshift = 10,
                )
                invalid.append(annotation)
        if invalid:
            fig.add_annotation(
                text = '<span style="color: #db0f0f"><b>!</b></span>  '
                       'Invalid Number of Beats ',
                align = 'right',
                showarrow = False,
                xref = 'paper',
                yref = 'paper',
                x = 1,
                y = 1.3)

        fig.update_layout(
            xaxis_title = 'Segment Number',
            xaxis = dict(tickmode = 'linear', dtick = 1),
            yaxis = dict(
                title = 'Number of Beats',
                range = [0, max_beats],
                dtick = dtick_value),
            legend = dict(
                orientation = 'h',
                yanchor = 'bottom',
                y = 1.0,
                xanchor = 'right',
                x = 1.0,
                traceorder = 'reversed'),
            annotations = invalid,
            font = dict(family = 'Poppins', size = 13),
            height = 289,
            margin = dict(t = 70, r = 20, l = 40, b = 65),
            barmode = 'overlay',
            template = 'simple_white',
        )
        if title is not None:
            fig.update_layout(
                title = title
            )
        return fig

    def compute_snr(
        self, 
        original: Union[np.ndarray, list], 
        filtered: Optional[Union[np.ndarray, list]] = None, 
        seg_size: int = 60, 
        powerline_freq: Literal[50, 60] = 60,
        signal_type: Literal['ECG', 'PPG'] = 'ECG',
        method: Literal['banded', 'residual', 'both'] = 'residual',
        combined_noise: bool = True,
        norm: bool = True
    ) -> pd.DataFrame:
        """
        Compute the signal-to-noise ratio of a cardiac signal by segment using
        power spectral density (PSD).
        
        Parameters
        ----------
        original : array-like
            The raw, unfiltered cardiac signal.
        filtered : array-like, optional
            The denoised signal after applying a filter. This array must be 
            given if `method` is set to 'residual'.
        seg_size : int, optional
            The size of the segment in seconds; by default, 60.
        powerline_freq : {50, 60}, optional
            The powerline interference frequency. This value must be either
            50 or 60 (by default, 60). The 50 Hz power grid is prevalent in
            many European, Asian, and African countries. This value must be 
            given if `method` is set to 'banded'.
        signal_type : {'ECG', 'PPG'}, optional
            The type of signal being processed to set signal and noise 
            frequency bands accordingly. This value must be given if `method` 
            is set to 'banded'.
            - 'ECG' (default): Uses 0.5–40 Hz as the ECG frequency band.
            - 'PPG': Uses 0.5–8 Hz as the PPG frequency band.
        method : {'banded', 'residual', 'both'}, optional
            The method used to estimate the noise component for SNR calculation.
            - 'banded': Uses PSD values within predefined frequency bands for 
              noise sources.
            - 'residual': Computes noise power as the PSD of the difference 
              between the original and filtered signals.
            - 'both': Computes both 'banded' and 'residual' SNR and returns 
              their average.
            By default, 'residual'.
        combined_noise : bool, optional
            If True, compute a single SNR value using the total power from 
            all noise sources. If False, compute separate SNR values for 
            baseline wander, motion artifacts, powerline interference, and 
            high-frequency noise. By default, True.
        norm : bool, optional
            Whether to normalize the SNR values; by default, True. Uses 
            min-max normalization.
    
        Returns
        -------
        SNR : pandas.DataFrame
            A DataFrame containing the SNR value(s) for each segment.
    
        Notes
        -----
        The choice of `method` affects how noise is defined and may influence 
        the resulting SNR values:
    
        - **'banded' method**: Assumes noise is confined to specific frequency 
          bands and does not significantly overlap with the signal. This is 
          useful when noise sources are known and well-separated in the 
          frequency domain.
        
        - **'residual' method**: Estimates noise dynamically by computing the 
          PSD of the difference between the original and filtered signals. 
          This approach captures all noise components, including those that 
          overlap with the signal band, making it more suitable for real-world 
          conditions where noise characteristics may vary.
        """
        
        # Check inputs
        if method not in ('banded', 'residual', 'both'):
            raise ValueError('method must be \'banded\', \'residual\', or '
                             '\'both\'.')
        if method in ('residual', 'both') and filtered is None:
            raise ValueError('filtered must be an array containing the '
                             'denoised signal.')
        if method == 'banded' and powerline_freq is None:
            raise ValueError('powerline_freq must be given when using '
                             '\'banded\' method.')
    
        def get_snr_resid(
            original: Union[np.ndarray, list], 
            filtered: Union[np.ndarray, list]
        ) -> float:
            """Compute the SNR with the residual approach."""
            freqs, psd_signal = welch(filtered, fs = self.fs, nperseg = 1024)
            _, psd_noise = welch(original - filtered, fs = self.fs, nperseg = 1024)
            snr_db = 10 * (np.log10(np.sum(psd_signal) / np.sum(psd_noise)))
            return snr_db
    
        def get_snr_banded(
            original: Union[np.ndarray, list], 
            signal_type: Literal['ECG', 'PPG'], 
            powerline_freq: Literal[50, 60], 
            combined_noise: bool
        ) -> float:
            """Compute the SNR with the banded approach."""
            
            # Define frequency bands
            pl_lower = powerline_freq - 1
            pl_upper = powerline_freq + 1
            if signal_type == 'ECG':
                signal_band = (0.5, 40)   # ECG signal band
                motion_band = (0.1, 5)    # Motion artifact band
                hf_cutoff = 40            # HF cutoff (includes EMG noise)
            elif signal_type == 'PPG':
                signal_band = (0.5, 8)    # PPG signal band
                motion_band = (0.1, 0.5)  # Motion artifact band
                hf_cutoff = 8             # HF cutoff
            else:
                raise ValueError('signal_type must be either \'ECG\' or \'PPG\'.')
                
            freqs, psd = welch(original, fs = self.fs, nperseg = 1024)
    
            # Get frequencies belonging to signal and noise sources
            signal_mask = (freqs >= signal_band[0]) & (freqs <= signal_band[1]) 
            bw_mask = (freqs < 0.5) 
            pl_mask = (freqs >= pl_lower) & (freqs <= pl_upper)
            motion_mask = (freqs >= motion_band[0]) & (freqs <= motion_band[1]) 
            hf_mask = (freqs > hf_cutoff) 
    
            # Compute power in signal and noise bands
            signal_power = np.sum(psd[signal_mask])
            if combined_noise:
                combined_noise_mask = bw_mask | pl_mask | motion_mask | hf_mask
                noise_power = np.sum(psd[combined_noise_mask])
                snr_db = 10 * np.log10(signal_power / max(noise_power, 1e-6))
            else:
                bw_power = np.sum(psd[bw_mask])
                pl_power = np.sum(psd[pl_mask])
                motion_power = np.sum(psd[motion_mask])
                hf_power = np.sum(psd[hf_mask])
                noise_power = [bw_power, pl_power, motion_power, hf_power]
                snr_db = [10 * np.log10(signal_power / max(p, 1e-6)) for p in noise_power]
                
            return snr_db
            
        seg_len = self.fs * seg_size
        n_seg = len(original) // seg_len
        snr_values = []
        
        for i in range(n_seg):
            if method == 'residual':
                seg_orig = original[(i * seg_len):((i + 1) * seg_len)]
                seg_filt = filtered[(i * seg_len):((i + 1) * seg_len)]
                snr_db = get_snr_resid(seg_orig, seg_filt)
            elif method == 'banded':
                segment = original[(i * seg_len):((i + 1) * seg_len)]
                snr_db = get_snr_banded(
                    segment, signal_type, powerline_freq, combined_noise)
            else:
                seg_orig = original[(i * seg_len):((i + 1) * seg_len)]
                seg_filt = filtered[(i * seg_len):((i + 1) * seg_len)]
                snr_resid = get_snr_resid(seg_orig, seg_filt)
                segment = original[(i * seg_len):((i + 1) * seg_len)]
                snr_banded = get_snr_banded(
                    segment, signal_type, powerline_freq, combined_noise)
                snr_db = np.mean([snr_resid, snr_banded])   
            snr_values.append(snr_db)
        
        if combined_noise:
            SNR = pd.DataFrame(snr_values, columns = ['SNR'])
        else:
            SNR = pd.DataFrame(snr_values, columns = [
                'Baseline Wander', 'Powerline Interference',
                'Motion Artifacts', 'HF Noise'])
        if norm:
            snr_min = SNR.min()
            snr_max = SNR.max()
            new_min = 0.01
            new_max = 1.0
            SNR = (SNR - snr_min) / (snr_max - snr_min) * (new_max - new_min) + new_min
        SNR.insert(0, 'Segment', np.arange(1, n_seg + 1))
        return SNR

    def _get_iqr(self, data: Union[np.ndarray, list]) -> float:
        """Compute the interquartile range of a data array."""
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        return iqr

    def _quartile_deviation(self, data: Union[np.ndarray, list]) -> float:
        """Compute the quartile deviation in the criterion beat difference
        test."""
        iqr = self._get_iqr(data)
        QD = iqr * 0.5
        return QD