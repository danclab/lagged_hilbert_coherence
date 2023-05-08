function amp_prods=arma_surr(signal, n_shuffles)

    n_trials = size(signal,1);
    n_pts = size(signal,2);

    amp_prods=zeros(n_shuffles*n_trials,1);
    pad=zeros(1,n_pts);

    for i=1:n_trials
        % Subtract out the mean and linear trend
        detrend_ord = 1;
        x=detrend(signal(i,:)-mean(signal(i,:)), detrend_ord);
        %x=signal(i,:);

        % Estimate an AR model
        mdl = arima(1,0,1);
        mdl = estimate(mdl, x', 'Display', 'off');

        for j=1:n_shuffles
            x_sim = simulate(mdl,n_pts)';

            % Subtract out the mean and linear trend
            x_sim=detrend(x_sim-mean(x_sim), detrend_ord);
        
            padd_rand_signal = [pad, x_sim, pad];
            % Get analytic signal (phase and amplitude)
            analytic_rand_signal = hilbert(padd_rand_signal);
            % Cut off padding
            analytic_rand_signal=analytic_rand_signal(:, n_pts+1:2 * n_pts);

            % Analytic signal at n=0...-1
            f1 = analytic_rand_signal(1:end-1);

            % Analytic signal at n=1...
            f2 = analytic_rand_signal(2:end);

            amp_prod = abs(f1) .* abs(f2);
            amp_prods((i-1)*n_shuffles+j)=mean(amp_prod);
        end
    end