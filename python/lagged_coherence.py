import math
import numpy as np
from joblib import Parallel, delayed
from scipy.signal import hilbert
from scipy.signal.windows import hann

import statsmodels.api as sm


def lagged_coherence(signal, freqs, lags, srate, win_size=3, type='coh', n_jobs=-1):
    """
    Compute lagged coherence (or phase-locking value or amplitude coherence) for a signal.

    Parameters
    ----------
    signal : ndarray
        The input signal, shape (n_trials, n_pts).
    freqs : array_like
        Frequencies of interest.
    lags : array_like
        Lags of interest.
    srate : float
        Sampling rate in Hz.
    win_size: float
        Size of the time window for each chunk in cycles (default = 3). If None, set to be equal to the evaluated lag.
    type : str
        Type of output: 'coh' for lagged coherence, 'plv' for lagged phase-locking value, or 'coh' for lagged amplitude
        coherence.
    n_jobs: integer
        The number of parallel jobs to run (default = -1). -1 means using all processors.

    Returns
    -------
    lcs : ndarray
        The output, shape (n_trials, n_freqs, n_lags).
    """

    # Number of trials
    n_trials = signal.shape[0]
    # Number of time points
    n_pts = signal.shape[1]

    # Number of frequencies
    n_freqs = len(freqs)
    # Number of lags
    n_lags = len(lags)

    # Create time
    T = n_pts * 1 / srate
    time = np.linspace(0, T, int(T * srate))

    def run_freq(f_idx):
        freq = freqs[f_idx]

        f_lcs = np.zeros((n_trials, n_lags))

        for l_idx, lag in enumerate(lags):

            # Width of time window to compute fourier coefficients in (cycles)
            if win_size is None:
                f_width = lag
            else:
                f_width = win_size

            # Width of time window in seconds
            width = f_width / freq

            # Half width
            halfwidth = width/2

            # Time steps
            start = time[0] + halfwidth
            stop = time[-1] - halfwidth
            step = lag/freq
            toi = np.arange(start, stop, step)

            # Initialize FFT coefficients - time step
            fft_coefs = np.zeros((n_trials, len(toi)), dtype=complex)
            for t_idx in range(len(toi)):
                # Chunk centered on time step
                chunk_start_time = toi[t_idx] - halfwidth
                chunk_stop_time = toi[t_idx] + halfwidth
                chunk_start = np.argmin(np.abs(time - chunk_start_time))
                chunk_stop = np.argmin(np.abs(time - chunk_stop_time))
                chunk = signal[:, chunk_start:chunk_stop]

                # Number of samples in chunk
                n_samps = chunk.shape[-1]

                # Hann windowing
                hann_window = hann(n_samps)
                hanned = chunk * hann_window

                # Get Fourier coefficients
                fourier_coef = np.fft.rfft(hanned)

                # Get frequencies from Fourier transformation
                fft_freqs = np.fft.rfftfreq(n_samps, d=1.0 / srate)

                # Find frequency closest to given
                fft_center_freq = np.argmin(np.abs(fft_freqs - freq))
                fft_coefs[:, t_idx] = fourier_coef[:, fft_center_freq]

            # Numerator is the sum of the fourier coefficients times the complex conjugate of the fourier coefficient
            # of the following chunk
            f1 = fft_coefs[:, :-1]
            f2 = fft_coefs[:, 1:]

            phase_diff = np.angle(f2) - np.angle(f1)
            amp_prod = np.abs(f1) * np.abs(f2)

            if type == 'coh':
                # Numerator - sum is over evaluation points
                num = np.sum(amp_prod * np.exp(complex(0, 1) * phase_diff), axis=-1)
                denom = np.sqrt(np.sum(np.abs(f1) ** 2, axis=-1) * np.sum(np.abs(f2) ** 2, axis=-1))
                lc = np.abs(num / denom)
            elif type == 'plv':
                expected_phase_diff = lag * 2 * math.pi
                num = np.sum(np.exp(complex(0, 1) * (expected_phase_diff - phase_diff)), axis=-1)
                denom = len(toi)
                lc = np.abs(num / denom)
            elif type == 'amp_coh':
                num = np.sum(amp_prod, axis=-1)
                denom = np.sqrt(np.sum(np.abs(f1) ** 2, axis=-1) * np.sum(np.abs(f2) ** 2, axis=-1))
                lc = num / denom
            f_lcs[:, l_idx] = lc
        return f_lcs

    lcs = Parallel(
        n_jobs=n_jobs
    )(delayed(run_freq)(f) for f in range(n_freqs))

    lcs = np.array(lcs)
    lcs = np.moveaxis(lcs, [0, 1, 2], [1, 0, 2])
    return lcs


def lagged_hilbert_coherence(signal, freqs, lags, srate, n_shuffles=1000, thresh_prctile=1, type='coh', n_jobs=-1):
    """
    Compute lagged Hilbert coherence (or phase-locking value or amplitude coherence) for a signal.

    Parameters
    ----------
    signal : ndarray
        The input signal, shape (n_trials, n_pts).
    freqs : array_like
        Frequencies of interest.
    lags : array_like
        Lags of interest.
    srate : float
        Sampling rate in Hz.
    type : str
        Type of output: 'coh' for lagged coherence, 'plv' for lagged phase-locking value, or 'coh' for lagged amplitude
        coherence.
    n_jobs: integer
        The number of parallel jobs to run (default = -1). -1 means using all processors.

    Returns
    -------
    lcs : ndarray
        The output, shape (n_trials, n_freqs, n_lags).
    """
    n_trials = signal.shape[0]
    n_pts = signal.shape[-1]
    T = n_pts * 1 / srate
    time = np.linspace(0, T, int(T * srate))

    # Number of frequencies
    n_freqs = len(freqs)
    # Number of lags
    n_lags = len(lags)

    # Bandpass filtering using multiplication by a Gaussian kernel
    # in the frequency domain
    # Frequency resolution
    df = np.diff(freqs)[0]

    def ar_surr(signal):
        n_trials = signal.shape[0]
        n_pts = signal.shape[-1]

        # Subtract out the mean and linear trend
        detrend_ord = 1
        x = sm.tsa.tsatools.detrend(signal, order=detrend_ord, axis=1)

        amp_prods = np.zeros(n_shuffles * n_trials)
        pad = np.zeros(n_pts)

        for i in range(n_trials):
            # Estimate an AR model
            mdl_order = (1, 0, 0)
            mdl = sm.tsa.ARIMA(x[i, :], order=mdl_order)
            result = mdl.fit()
            # Make a generative model using the AR parameters
            arma_process = sm.tsa.ArmaProcess.from_coeffs(result.arparams)
            # Simulate a bunch of time-courses from the model
            x_sim = arma_process.generate_sample((n_pts, n_shuffles),
                                                 scale=result.resid.std())
            # Subtract out the mean and linear trend
            x_sim = sm.tsa.tsatools.detrend(x_sim, order=detrend_ord, axis=0)

            for j in range(n_shuffles):
                padd_rand_signal = np.hstack([pad, x_sim[:, j], pad])
                # Get analytic signal (phase and amplitude)
                analytic_rand_signal = hilbert(padd_rand_signal, N=None)[n_pts:2 * n_pts]

                # Analytic signal at n=0...-1
                f1 = analytic_rand_signal[0:-1]
                # Analytic signal at n=1,...
                f2 = analytic_rand_signal[1:]

                amp_prod = np.abs(f1) * np.abs(f2)
                amp_prods[i * n_shuffles + j] = np.mean(amp_prod[:])

        return amp_prods

    # Compute threshold as 5th percentile of shuffled amplitude
    # products
    amp_prods = ar_surr(signal)
    thresh = np.percentile(amp_prods, thresh_prctile)

    padd_signal = np.hstack([np.zeros((n_trials, n_pts)), signal, np.zeros((n_trials, n_pts))])
    signal_fft = np.fft.rfft(padd_signal, axis=-1)
    fft_frex = np.fft.rfftfreq(padd_signal.shape[-1], d=1 / srate)
    sigma = df * .5

    def run_freq(f_idx):
        freq = freqs[f_idx]

        f_lcs = np.zeros((n_trials, n_lags))

        # Gaussian kernel centered on frequency with width defined
        # by requested frequency resolution
        kernel = np.exp(-((fft_frex - freq) ** 2 / (2.0 * sigma ** 2)))

        # Multiply Fourier-transformed signal by kernel
        fsignal_fft = np.multiply(signal_fft, kernel)
        # Reverse Fourier to get bandpass filtered signal
        f_signal = np.fft.irfft(fsignal_fft, axis=-1)

        # Get analytic signal of bandpass filtered data (phase and amplitude)
        analytic_signal = hilbert(f_signal, N=None, axis=-1)
        # Cut off padding
        analytic_signal = analytic_signal[:, len(time):2 * len(time)]

        for l_idx, lag in enumerate(lags):
            # Duration of this lag in s
            lag_dur_s = lag / freq
            # Number of evaluations
            n_evals = int(np.floor(T / lag_dur_s))
            # Remaining time
            diff = T - (n_evals * lag_dur_s)

            # Start time
            start_time = time[0]
            # Evaluation times (ms)
            eval_times = np.linspace(start_time, T - diff, n_evals + 1)[:-1]
            # Evaluation time points
            eval_pts = np.searchsorted(time, eval_times)

            # This was evaluated starting at time t=0 and looking 2 cycles ahead,
            # but what about the points in between?

            # Number of points between the first and next evaluation time points
            n_range = eval_pts[1] - eval_pts[0]
            # Analytic signal at n=0...n_evals-1 evaluation points, and m=0..n_range time points in between
            f1 = analytic_signal[:, eval_pts[:-1, np.newaxis] + np.arange(n_range)][:, :, np.newaxis]
            # Analytic signal at n=1...n_evals evaluation points, and m=0..n_range time points in between
            f2 = analytic_signal[:, eval_pts[1:, np.newaxis] + np.arange(n_range)][:, :, np.newaxis]

            # Calculate the phase difference and amplitude product
            phase_diff = np.angle(f2) - np.angle(f1)
            amp_prod = np.abs(f1) * np.abs(f2)

            if type == 'coh':
                # Lagged coherence
                num = np.squeeze(np.sum(amp_prod * np.exp(complex(0, 1) * phase_diff), axis=1))
                f1_pow = np.power(f1, 2)
                f2_pow = np.power(f2, 2)
                denom = np.squeeze(np.sqrt(np.sum(np.abs(f1_pow), axis=1) * np.sum(np.abs(f2_pow), axis=1)))
                lc = np.abs(num / denom)
                lc[denom < thresh] = 0

            elif type == 'plv':
                expected_phase_diff = lag * 2 * math.pi
                num = np.squeeze(np.sum(np.exp(complex(0, 1) * (expected_phase_diff - phase_diff)), axis=1))
                denom = len(eval_pts) - 1
                lc = np.abs(num / denom)

                f1_pow = np.power(f1, 2)
                f2_pow = np.power(f2, 2)
                amp_denom = np.squeeze(np.sqrt(np.sum(np.abs(f1_pow), axis=1) * np.sum(np.abs(f2_pow), axis=1)))
                # Threshold based on amplitude denominator
                lc[amp_denom < thresh] = 0

            elif type == 'amp_coh':
                # Numerator - sum is over evaluation points
                num = np.squeeze(np.sum(amp_prod, axis=1))
                f1_pow = np.power(f1, 2)
                f2_pow = np.power(f2, 2)
                denom = np.squeeze(np.sqrt(np.sum(np.abs(f1_pow), axis=1) * np.sum(np.abs(f2_pow), axis=1)))
                # Calculate LC
                lc = np.abs(num / denom)
                # Threshold based on denominator
                lc[denom < thresh] = 0

            # Average over the time points in between evaluation points
            f_lcs[:, l_idx] = np.mean(lc, axis=-1)

        return f_lcs

    lcs = Parallel(
        n_jobs=n_jobs
    )(delayed(run_freq)(f) for f in range(n_freqs))

    lcs = np.array(lcs)
    lcs = np.moveaxis(lcs, [0, 1, 2], [1, 0, 2])
    return lcs