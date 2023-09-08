function amp_prods=ar_surr(signal, n_shuffles)

    n_trials = size(signal, 1);
    n_pts = size(signal, 2);

    amp_prods = zeros(n_trials, n_shuffles, n_pts-1);

    parfor i = 1:n_trials
        x = signal(i, :);

        % Estimate an AR model
        mdl = arima(1, 0, 0);
        mdl = estimate(mdl, x', 'Display', 'off');

        x_sim = simulate(mdl, n_pts, 'NumPaths', n_shuffles)';

        padd_rand_signal = [zeros(size(x_sim)), x_sim, zeros(size(x_sim))];
        analytic_rand_signal = hilbert(padd_rand_signal')';
        analytic_rand_signal = analytic_rand_signal(:, n_pts+1:2*n_pts);

        f1 = analytic_rand_signal(:, 1:end-1);
        f2 = analytic_rand_signal(:, 2:end);
        amp_prods(i, :, :) = abs(f1) .* abs(f2);

    end