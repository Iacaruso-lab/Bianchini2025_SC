
function U_events = getUEvents(e, mod, which_trials)
    if mod == 1
        U_events = e(:, 1, which_trials(end - 1));
    elseif mod == 2
        U_events = e(:, 1, which_trials(end));
    end
end