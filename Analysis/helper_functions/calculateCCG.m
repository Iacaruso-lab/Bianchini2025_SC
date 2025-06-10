function CCG = computeCCG(x1, x2, M, N, tau, lambda1, lambda2)
    numerator = 0;
    for i = 1:M
        for t = 1:N-tau
            numerator = numerator + x1(i, t) * x2(i, t+tau);
        end
    end
    numerator = numerator / M;

    denominator = sqrt(lambda1 * lambda2);

    CCG = numerator / (denominator * theta(tau, N));
end

function result = theta(tau, N)
    if abs(tau) <= N
        result = N - abs(tau);
    else
        result = 0;
    end
end

% Example usage
M = 100;  % Number of trials
N = 1000;  % Number of time bins
x1 = randi([0, 1], M, N);  % Spike trains of neuron j
x2 = randi([0, 1], M, N);  % Spike trains of neuron k
tau = 10;  % Time lag
lambda1 = 2;  % Mean firing rate of neuron j
lambda2 = 3;  % Mean firing rate of neuron k

CCG = computeCCG(x1, x2, M, N, tau, lambda1, lambda2);
disp(CCG);
