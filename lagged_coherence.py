import numpy as np
from joblib import Parallel, delayed
from scipy.signal import hilbert
from scipy.signal.windows import hann


def lagged_coherence_classic(signal, freqs, lags, srate, type='coh'):
    # Number of trials
    n_trials = signal.shape[0]
    # Number of frequencies
    n_freqs = len(freqs)
    # Number of lags
    n_lags = len(lags)

    # Create time
    n_pts = signal.shape[1]
    T = n_pts * 1 / srate
    time = np.linspace(0, T, int(T * srate))

    # Frequency resolution
    df = np.diff(freqs)[0]
    # Required sample size for this frequency resolution
    req_fft_size = int(srate / df)

    def run_freq(f_idx):
        freq = freqs[f_idx]

        f_lcs = np.zeros((n_trials, n_lags))
        for l_idx, lag in enumerate(lags):
            # Width of time window to compute fourier coefficients in (cycles)
            f_width = lag

            # Width of time window in seconds
            width = f_width / freq

            # Half width
            halfwidth = np.ceil(srate * width / 2) / srate

            # Time steps
            start = time[0] + halfwidth
            stop = time[-1] - halfwidth
            step = np.ceil(srate * lag / freq) / srate
            toi = np.arange(start, stop, step)

            # Initialize FFT coefficients - trial x time step
            fft_coefs = np.zeros((n_trials, len(toi)), dtype=complex)
            for t_idx in range(len(toi)):
                # Chunk centered on time step
                chunk_start_time = toi[t_idx] - halfwidth
                chunk_stop_time = toi[t_idx] + halfwidth
                chunk_start = np.argmin(np.abs(time - chunk_start_time))
                chunk_stop = np.argmin(np.abs(time - chunk_stop_time))
                chunk = signal[:, chunk_start:chunk_stop]

                # Number of samples in chunk
                n_samps = chunk.shape[1]

                # Size of padding needed for frequency resolution
                pad_size = np.max([0, req_fft_size - n_samps])

                # Zero-padding
                padd_chunk = np.hstack([np.zeros((n_trials, int(pad_size / 2))),
                                        chunk,
                                        np.zeros((n_trials, int(pad_size / 2)))])

                # Number of samples in zero-padded chunk
                n_padd_samps = padd_chunk.shape[1]

                # Hann windowing
                hann_window = hann(n_padd_samps)
                hanned = padd_chunk * hann_window

                # Get Fourier coefficients
                fourier_coef = np.fft.rfft(hanned)

                # Get frequencies from Fourier transformation
                fft_freqs = np.fft.rfftfreq(n_padd_samps, d=1.0 / srate)

                # Average Fourier components within window +/- 1Hz around frequency
                fft_center_freq = np.argmin(np.abs(fft_freqs - freq))
                # fft_freqs_idx = [fft_center_freq - 1, fft_center_freq + 1]
                # fft_coefs[:, t_idx] = np.mean(fourier_coef[:, fft_freqs_idx[0]:fft_freqs_idx[1]], axis=1)
                fft_coefs[:, t_idx] = fourier_coef[:, fft_center_freq]

            # Numerator is the sum of the fourier coefficients times the complex conjugate of the fourier coefficient
            # of the following chunk
            f1 = fft_coefs[:, :-1]
            f2 = fft_coefs[:, 1:]

            phase_diff = np.angle(f2) - np.angle(f1)
            amp_prod = np.abs(f1) * np.abs(f2)

            if type == 'coh':
                # Numerator - sum is over evaluation points
                num = np.sum(amp_prod * np.exp(complex(0, 1) * phase_diff), axis=1)
                denom = np.sqrt(np.sum(np.abs(f1) ** 2, axis=-1) * np.sum(np.abs(f2) ** 2, axis=-1))
                lc = np.abs(num / denom)
            elif type=='plv':
                num = np.sum(np.exp(complex(1, 0) * phase_diff), axis=1)
                denom = len(toi)
                lc = np.abs(num / denom)
            elif type=='amp_coh':
                num = np.sum(amp_prod, axis=-1)
                denom = np.sqrt(np.sum(np.abs(f1) ** 2, axis=-1) * np.sum(np.abs(f2) ** 2, axis=-1))
                lc = num / denom

            f_lcs[:, l_idx] = lc
        return f_lcs

    lcs = Parallel(
        n_jobs=-1
    )(delayed(run_freq)(f) for f in range(n_freqs))

    lcs = np.array(lcs)
    lcs = np.moveaxis(lcs, [0, 1, 2], [1, 0, 2])

    return lcs


def phase_shuffle(signal):
    # Number of trials
    n_trials = signal.shape[0]
    # Create time
    n_pts = signal.shape[1]

    # Pre-allocate memory for shuffled_matrix
    shuffled_matrix = np.zeros((n_trials, n_pts))

    # Fourier transform of matrix
    ts_fourier = np.fft.rfft(signal, axis=-1)

    # Generate random phases
    random_phases = np.exp(np.random.uniform(0, np.pi, ts_fourier.shape) * 1.0j)

    # Apply random phases to Fourier transform
    ts_fourier_new = ts_fourier * random_phases

    # Inverse Fourier transform to get shuffled matrix
    shuffled_matrix = np.fft.irfft(ts_fourier_new, axis=-1)
    return shuffled_matrix


def lagged_surrogate_coherence(signal, freqs, lags, srate, n_shuffles=1000, thresh_prctile=1, type='coh'):
    # Number of trials
    n_trials = signal.shape[0]
    # Number of frequencies
    n_freqs = len(freqs)
    # Number of lags
    n_lags = len(lags)

    # Create time
    n_pts = signal.shape[1]
    T = n_pts * 1 / srate
    time = np.linspace(0, T, int(T * srate))

    # Frequency resolution
    df = np.diff(freqs)[0]

    def run_shuffle():
        rand_signal = phase_shuffle(signal)
        padd_rand_signal = np.hstack([np.zeros((n_trials, n_pts)), rand_signal, np.zeros((n_trials, n_pts))])
        # Get analytic signal (phase and amplitude)
        analytic_signal = hilbert(padd_rand_signal, N=None, axis=-1)[:, n_pts:2 * n_pts]

        # Analytic signal at n=0...-1
        f1 = analytic_signal[:, 0:-1]
        # Analytic signal at n=1,...
        f2 = analytic_signal[:, 1:]

        amp_prod = np.abs(f1) * np.abs(f2)
        return np.mean(amp_prod, axis=0)

    amp_prods = Parallel(
        n_jobs=-1
    )(delayed(run_shuffle)() for i in range(n_shuffles))
    amp_prods = np.array(amp_prods)
    thresh = np.percentile(amp_prods[:], thresh_prctile)

    padd_signal = np.hstack([np.zeros((n_trials, n_pts)), signal, np.zeros((n_trials, n_pts))])
    signal_fft = np.fft.rfft(padd_signal, axis=-1)
    fft_frex = np.fft.rfftfreq(padd_signal.shape[-1], d=1 / srate)
    sigma = df * .5

    def run_freq(f_idx):
        freq = freqs[f_idx]

        f_lcs = np.zeros((n_trials, n_lags))

        kernel = np.exp(-((fft_frex - freq) ** 2 / (2.0 * sigma ** 2)))

        fsignal_fft = np.multiply(signal_fft, kernel)
        f_signal = np.fft.irfft(fsignal_fft, axis=-1)
        # Get analytic signal (phase and amplitude)
        analytic_signal = hilbert(f_signal, N=None, axis=-1)[:, n_pts:2 * n_pts]

        for l_idx, lag in enumerate(lags):
            # Duration of this lag in s
            lag_dur_ms = lag * 1 / freq
            # Number of evaluation
            n_evals = int(np.floor(T / lag_dur_ms))
            diff = T - (n_evals * lag_dur_ms)

            # Start time
            start_time = time[0]
            # Evaluation times (ms)
            eval_times = np.linspace(start_time, T - diff, n_evals + 1)
            # Evaluation time points
            eval_pts = np.searchsorted(time, eval_times)[:-1]

            # Number of points between the first and next evaluation time points
            n_range = eval_pts[1] - eval_pts[0]
            # Analytic signal at m=0...n_evals-1 evaluation points, and n=0..n_range time points in between
            f1 = analytic_signal[:, eval_pts[:-1, np.newaxis] + np.arange(n_range)[np.newaxis, :]]
            # Analytic signal at m=1...n_evals evaluation points, and n=0..n_range time points in between
            f2 = analytic_signal[:, eval_pts[1:, np.newaxis] + np.arange(n_range)[np.newaxis, :]]
            # calculate the phase difference and amplitude product
            phase_diff = np.angle(f2) - np.angle(f1)
            amp_prod = np.abs(f1) * np.abs(f2)
            if type=='coh':
                # Numerator - sum is over evaluation points
                num = np.sum(amp_prod * np.exp(complex(0, 1) * phase_diff), axis=1)
                # Scaling factor - sum is over evaluation points
                denom = np.sqrt(np.sum(np.abs(np.power(f1, 2)), axis=1) * np.sum(np.abs(np.power(f2, 2)), axis=1))
                # Calculate LC
                range_lcs = np.abs(num / denom)
                # Threshold based on denominator
                range_lcs[denom < thresh] = 0
            elif type=='plv':
                expected_phase_diff=lag*2*math.pi
                num = np.sum(np.exp(complex(1, 0) * (expected_phase_diff-phase_diff)), axis=1)
                denom = len(eval_pts)
                range_lcs = np.abs(num / denom)
                amp_denom = np.sqrt(np.sum(np.abs(np.power(f1, 2)), axis=1) * np.sum(np.abs(np.power(f2, 2)), axis=1))
                # Threshold based on amplitude denominator
                range_lcs[amp_denom < thresh] = 0
            elif type=='amp_coh':
                # Numerator - sum is over evaluation points
                num = np.sum(amp_prod, axis=1)
                # Scaling factor - sum is over evaluation points
                denom = np.sqrt(np.sum(np.abs(np.power(f1, 2)), axis=1) * np.sum(np.abs(np.power(f2, 2)), axis=1))
                # Calculate LC
                range_lcs = np.abs(num / denom)
                # Threshold based on denominator
                range_lcs[denom < thresh] = 0
            # Average over time points in between evaluation points
            f_lcs[:, l_idx] = np.mean(range_lcs, axis=1)
        return f_lcs

    lcs = Parallel(
        n_jobs=-1
    )(delayed(run_freq)(f) for f in range(n_freqs))

    lcs = np.array(lcs)
    lcs = np.moveaxis(lcs, [0, 1, 2], [1, 0, 2])

    return lcs