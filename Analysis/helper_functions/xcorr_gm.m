function [c_jk_norm, c_jk_unnorm] = xcorr_gm(pre, post, maxlag, minlag)
% input
%   pre: num_trials x num_timepts logical array of presyn neuron spikes
%   post: num_trials x num_timepts logical array of postsyn neuron spikes
%   maxlag: number of trials in past to compute ccg, e.g. 25
%   minlag: number of trials in future to compute ccg, e.g. 0
% output
%   c_jk

c_jk_norm = zeros(1,maxlag-minlag+1,'single');
c_jk_unnorm= zeros(1,maxlag-minlag+1,'single');
for lag=-maxlag:1:-minlag
    if lag<=0
        prev = pre(:,1:end+lag);
        postv = post(:,1-lag:end);
    else
        prev = pre(:,1+lag:end);
        postv = post(:,1:end-lag);
    end
    c_jk_unnorm(lag+maxlag+1) = sum(prev .* postv, 'all');
    c_jk_norm(lag+maxlag+1) = c_jk_unnorm(lag+maxlag+1)/sqrt(sum(prev, 'all')*sum(postv, 'all'));
end

end
