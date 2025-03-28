function which_trials = getWhichTrials(e)
    n_trials = length(e(1, 1, :));
    if n_trials == 16
        which_trials = [1:11, 15, 16];
    elseif n_trials == 19
        which_trials = [1:11, 18, 19];
    elseif n_trials == 14
        which_trials = [1:13];
    end
end