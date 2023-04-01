function [num,denom,lc]=lagged_hilbert_coherence(signal, srate, freq, lag)
    
    n_pts = length(signal);
    dt=1/srate;
    T = n_pts * dt;
    time=0:dt:T-dt;
    
    % Bandpass filtering using multiplication by a Gaussian kernel
    % in the frequency domain

    % Frequencies to evaluate (just to compute desired frequency
    % resolution in this step)
    freqs=linspace(5,100,100);

    % Determine the frequency resolution
    df = diff(freqs(1:2));

    % Zero-pad the signal - or is mean-pad better?
    pad=zeros(size(time));
    padd_signal = [pad, signal, pad];

    % Fourier transform the padded signal
    signal_fft = rfft(padd_signal,length(padd_signal), 2);
    fft_frex = linspace(0, srate/2, length(signal_fft));

    % Kernel width for multiplication
    sigma = df * .5;

    % Gaussian kernel centered on frequency with width defined
    % by requested frequency resolution
    kernel = exp(-((fft_frex - freq) .^ 2 / (2.0 * sigma ^ 2)));

    % Multiply Fourier-transformed signal by kernel
    fsignal_fft = signal_fft .* kernel;
    % Reverse Fourier to get bandpass filtered signal
    f_signal = irfft(fsignal_fft, length(padd_signal), 2);
    
    % Get analytic signal of bandpass filtered data (phase and amplitude)
    analytic_signal = hilbert(f_signal);
    % Cut off padding
    analytic_signal=analytic_signal(:, length(time)+1:2 * length(time));
    
    % Duration of this lag in s
    lag_dur_s = lag / freq;
    % Number of evaluations
    n_evals = floor(T / lag_dur_s);
    % Remaining time
    t_diff = T - (n_evals * lag_dur_s);

    % Start time
    start_time = time(1);
    % Evaluation times (ms)
    eval_times = linspace(start_time, T - t_diff, n_evals + 1); 
    eval_times = eval_times(1:end-1);
    % Evaluation time points
    eval_pts = knnsearch(time', eval_times');
    
    % This was evaluated starting at time t=0 and looking 2 cycles ahead, 
    % but what about the points in between?

    % Number of points between the first and next evaluation time points
    n_range = eval_pts(2) - eval_pts(1);

    % Analytic signal at n=0...n_evals-1 evaluation points, and m=0..n_range time points in between
    f1 = analytic_signal(eval_pts(1:end-1) + (0:n_range-1));

    % Analytic signal at n=1...n_evals evaluation points, and m=0..n_range time points in between
    f2 = analytic_signal(eval_pts(2:end) + (0:n_range-1));

    % Calculate the phase difference and amplitude product
    phase_diff = angle(f2) - angle(f1);
    amp_prod = abs(f1) .* abs(f2);

    % Lagged coherence
    num = abs(sum(amp_prod .* exp(1i * phase_diff),1));
    f1_pow = f1.^2;
    f2_pow = f2.^2;
    denom = sqrt(sum(abs(f1_pow),1) .* sum(abs(f2_pow),1));
    lc = num / denom;

    % Average over the time points in between evaluation points
    lc = mean(lc);