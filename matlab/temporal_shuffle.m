% Define temporal shuffle function
function [amp_prod_mean] = temporal_shuffle(signal)
    % Shuffle the signal
    rand_signal = signal(randperm(length(signal)));
    pad = zeros(size(signal));
    padd_rand_signal = [pad, rand_signal, pad];
    
    % Get analytic signal (phase and amplitude)
    analytic_rand_signal = hilbert(padd_rand_signal);
    analytic_rand_signal=analytic_rand_signal(:, length(signal)+1:2 * length(signal));

    % Analytic signal at n=0...-1
    f1 = analytic_rand_signal(1:end-1);
    % Analytic signal at n=1,...
    f2 = analytic_rand_signal(2:end);

    % Compute amplitude products and return mean
    amp_prod = abs(f1) .* abs(f2);
    amp_prod_mean = mean(amp_prod);
end

