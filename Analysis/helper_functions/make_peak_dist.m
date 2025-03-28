function [peak_dist] = make_peak_dist(d,s,e,n_perm)
%creates a distribution of possible sums based on the peak FR

% parameters
tr_n = 11;
db=0.25; % duration
window_res = 250;
smooth=0.5; % smoothing factor
dt_ms = 1;
line=linspace(0,db+0.01,1/(dt_ms*10^-3));

% get the trials
which_trials = getWhichTrials(e);

% select which modality
for mod = 1:2

    % select which modality to compare it to
    U_events = getUEvents(e, mod, which_trials);

    for t = 1:tr_n
        pooled_trials = [e(:,1,t);U_events]; %pool together trials

        % bootstrap approach
        list_of_peak_heights=zeros(n_perm,1);

        for perm = 1:n_perm
            idx=randperm(length(pooled_trials),50);
            not_idx=setdiff(1:100,idx);
            sampleA=pooled_trials(idx);
            sampleB=pooled_trials(not_idx);

            [~,~,sIFR1]=getIFR(s, sampleA,db,smooth);
            [~,~,sIFR2]=getIFR(s,sampleB,db,smooth);

            % I need to interpolate them because I need the time to be
            % the same between the two
            a = 0;
            b = 0.001;
            r = (b-a).*rand(10000,1) + a;

            % if there are duplicates in time adjust for it
            [~, w] = unique( sIFR1.vecTime, 'stable' );
            duplicate_indices = setdiff( 1:numel(sIFR1.vecTime), w );
            sIFR1.vecTime(duplicate_indices)=sIFR1.vecTime(duplicate_indices)+r(duplicate_indices);

            [~, w] = unique(sIFR2.vecTime, 'stable' );
            duplicate_indices = setdiff( 1:numel(sIFR2.vecTime), w );
            sIFR2.vecTime(duplicate_indices)=sIFR2.vecTime(duplicate_indices)+r(duplicate_indices);

            %perform interpolation
            tr1=interp1(sIFR1.vecTime,sIFR1.vecRate,line,'linear');
            tr2=interp1(sIFR2.vecTime,sIFR2.vecRate,line,'linear');

            % find the max value for each
            [~,pos]=max(tr1);
            if pos<6
                max_tr1=mean(tr1(1:pos+5));
            elseif pos>6
                max_tr1=mean(tr1(pos-5:pos+5));
            end

            [~,pos]=max(tr2);
            if pos<6
                max_tr2=mean(tr2(1:pos+5));
            elseif pos>6
                max_tr2=mean(tr2(pos-5:pos+5));
            end
            list_of_peak_heights(perm)=max_tr1-max_tr2;% put them together in a list

        end
        peak_dist(mod,t).max = list_of_peak_heights;
        t
    end

end
