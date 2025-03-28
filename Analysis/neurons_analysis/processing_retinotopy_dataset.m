%% load the data

close all
clear all
clc

on_cluster = 0;
which_dataset = 'retinotopy_preprocessing'; % let's select the dataset
if on_cluster == 1
    file_path = fullfile('/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/paper_code/Datasets/neurons_datasets/pre_processing_datasets', [which_dataset '.mat']);
    addpath(genpath('/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/spikes-master')) % for converting to Phy
    addpath(genpath('/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/ZETA')) % for zeta testing
else
    file_path = fullfile('Z:\home\shared\Gaia\Coliseum\Delays\paper_code\Datasets\neurons_datasets\pre_processing_datasets', [which_dataset '.mat']);
end

load(file_path)

%% details of stimuli

n_col = 7;
n_rows = 5;
locs = n_col*n_rows;
stims = [1,locs; locs+1,locs*2; locs*2+1,locs*3];

%% save slices positions

recs = unique(extractfield(neuron_matrix,'Experiment_ID'));

for i = 1:length(recs)
    ix = recs(i);
    pos = find(extractfield(neuron_matrix,'Experiment_ID')==ix);
    
    for p = 1:length(pos)
        px = pos(p);

        if ix == 82 || ix == 111 || ix == 113
            neuron_matrix(px).slices = [3:9];
            neuron_matrix(px).slices_degrees =[-126:18:-18];
        elseif ix == 92
             neuron_matrix(px).slices = [4:10];
             neuron_matrix(px).slices_degrees = [-108:18:0];
        else
             neuron_matrix(px).slices = [1:7];
             neuron_matrix(px).slices_degrees = [-162:18:-54];
        end

    end
end

%% Calculate IFR

for d=1:length(neuron_matrix)
   
    s = neuron_matrix(d).spikeTimes;
    e = neuron_matrix(d).eventTimes -0.3;
    db=1;
    intSmoothSd=1;
    dt_ms = 1;
    line=linspace(0,1,1/(dt_ms*10^-3));   
    
    smooth_FR=zeros(length(e),1000);
    
    for i=1:length(e)
        events = e(:,1,i);
        [~,~,sIFR] = getIFR(s,events,db,intSmoothSd);
        
        %correct for duplicates
        a = 0;
        b = 0.001;
        r = (b-a).*rand(10000,1) + a;
        [v, w] = unique( sIFR.vecTime, 'stable' );
        duplicate_indices = setdiff( 1:numel(sIFR.vecTime), w );
        sIFR.vecTime(duplicate_indices)=sIFR.vecTime(duplicate_indices)+r(duplicate_indices);
        
        [v, w] = unique(sIFR.vecRate, 'stable' );
        duplicate_indices = setdiff( 1:numel(sIFR.vecRate), w );
        sIFR.vecRate(duplicate_indices)=sIFR.vecRate(duplicate_indices)+r(duplicate_indices);
        %perform interpolation
        interp_result=interp1(sIFR.vecTime,sIFR.vecRate,line,'linear');
        
        smooth_FR(i,:) = movmean(interp_result,10);
    end
    IFR(d).mean=smooth_FR;
    d
end

%% Save the IFR separately

file_path2 = fullfile('/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/save_load_datasets', ['IFR_retinotopy' '.mat']);
save(file_path2,'IFR','-v7.3')

%% correct for multiple comparison and save the output

for d = 1:length(neuron_matrix)

    Z_pval = neuron_matrix(d).Z_pval;
    IFR_ret = IFR(d).mean;
    maxf  =max(IFR_ret(:));

    AUD= pval_adjust(Z_pval(stims(1,1):stims(1,2)),'BH');
    if any(AUD<0.05) && maxf>=10
        neuron_matrix(d).AUD = 1;
    else
        neuron_matrix(d).AUD = 0;
    end

    VIS = pval_adjust(Z_pval(stims(2,1):stims(2,2)),'BH');
    if any(VIS<0.05) && maxf>=10
        neuron_matrix(d).VIS = 1;
    else
        neuron_matrix(d).VIS = 0;
    end

    MULTI =pval_adjust( Z_pval(stims(3,1):stims(3,2)),'BH');
    if neuron_matrix(d).AUD == 1 && neuron_matrix(d).VIS ==1 && maxf>=10
        neuron_matrix(d).AV = 1;
    else
        neuron_matrix(d).AV = 0;
    end

end

%% Let's save the mean response profile for each neruon as well as the max FR

for d=1:length(neuron_matrix)

    s = neuron_matrix(d).spikeTimes;
    e = neuron_matrix(d).eventTimes; 

    binSize=0.01; %binsize is of 10 ms
    win=[0 0.15]; %the first bin will be (-0.5 -0.1)

    meanFR=zeros(size(e,3),1);
    for g=1:size(e,3)
        [psth, bins, rasterX, rasterY, spikeCounts, ba] = psthAndBA(s, e(:,1,g), win, binSize);
        meanFR(g)=mean(spikeCounts);
    end

    neuron_matrix(d).AUD_ret = meanFR(stims(1,1):stims(1,2));
    neuron_matrix(d).VIS_ret = meanFR(stims(2,1):stims(2,2));
    neuron_matrix(d).MULTI_ret = meanFR(stims(3,1):stims(3,2));

    % let's look at the max FR
    IFR_ret = IFR(d).mean;
    FR = zeros(size(IFR_ret,1),1);
    for tr = 1:size(IFR_ret,1)
            curr_IFR = IFR_ret(tr,290:end-211);
            [val,pos]=max(curr_IFR);
            FR(tr) = mean(IFR_ret(tr,(290+pos)-10:(290+pos)+10));
    end
    neuron_matrix(d).AUD_peaks = FR(stims(1,1):stims(1,2));
    neuron_matrix(d).VIS_peaks = FR(stims(2,1):stims(2,2));
    neuron_matrix(d).MULTI_peaks = FR(stims(3,1):stims(3,2));
end

%% now do the gaussian fit to get an estimate of the RF size 

n_mod = 3;
for d=1:length(neuron_matrix)
    d
    VIS_ret = neuron_matrix(d).VIS_peaks;
    AUD_ret = neuron_matrix(d).AUD_peaks;
    MULTI_ret = neuron_matrix(d).MULTI_peaks;

    fit_params = zeros(7,1);
    fits = zeros(n_col,n_rows);
    fits_interpolated = zeros(90,126);%n_col*10,n_rows*10);

    for mod = 1:n_mod
        if mod == 1
            RF = AUD_ret;
        elseif mod == 2
            RF = VIS_ret;
        else
            RF = MULTI_ret;
        end

        %normalize data between 0 and 1
        norm_data = (RF - nanmin(RF)) / (nanmax(RF) - nanmin(RF));
        norm_data = reshape(norm_data,n_rows,n_col);

        % Replace NaN with zero
        nanIndices = isnan(norm_data);
        norm_data(nanIndices) = 0;
        nX = n_rows;
        nY = n_col;
        xx = 1:nX;
        yy = 1:nY;

        [fitresult, zfit, fiterr, zerr, resnorm, rr] = fmgaussfit(xx,yy,norm_data);
        fit_params = fitresult;

        fit = zfit;
        thisMean = repmat(mean(reshape(norm_data,[nX*nY,1])),nX,nY);
        SS_res = sum(reshape(norm_data - fit,[nX*nY,1]).^2);
        SS_tot = sum(reshape(norm_data - thisMean,[nX*nY,1]).^2);
        r_squared = 1 - (SS_res/SS_tot);

        % get interpolated fitted data
        xVals = (1:0.01:nX);
        yVals = (1:0.01:nY);

        zVals = ones(length(xVals),length(yVals));
        [xData, yData, zData] = prepareSurfaceData(xVals, yVals, zVals);
        xyData = {xData,yData};

        z = gaussian2D(fitresult,xyData);
        fits_interpolated = reshape(z,[length(xVals),length(yVals)]);

        % now measure the area
        size_matrix = size(fits_interpolated);
        a2 = ones(size_matrix);
        tot_area = bwarea(a2);
        x_ret = 126; % 18degreees x 7 slices
        y_ret = 90; %18degrees x 5 slices
        screen_area = x_ret * y_ret; % x:47.6, y: 26.7
        conv_factor = screen_area/tot_area;

        actual = fits_interpolated;

        % Threshold to binary mask using half peak
        half_peak = (max(actual(:)) - min(actual(:))) / 1.3;
        half_peak = half_peak + min(actual(:));
        actual(actual < half_peak) = 0;
        actual(actual >= half_peak) = 1;

        % Calculate the area of the binary mask
        % lets' find the radius in x and conver it
        x_indices = find(actual == 1);

        % Convert the linear indices to subscripts
        [x_sub, y_sub] = ind2sub(size(actual), x_indices);
        x_rad =  (max(x_sub)-min(x_sub))/2;
        x_angle = (x_rad*x_ret)/ size_matrix(1);
        y_rad = (max(y_sub)-min(y_sub))/2;
        y_angle = (y_rad*y_ret)/ size_matrix(2);
        RF_area = pi*x_angle*y_angle;

        % save it
        if mod == 1
            neuron_matrix(d).AUD_RF_area = RF_area;
            neuron_matrix(d).AUD_r_squared = r_squared;
            neuron_matrix(d).AUD_fits_interp = fits_interpolated;
        elseif mod == 2
            neuron_matrix(d).VIS_RF_area = RF_area;
            neuron_matrix(d).VIS_r_squared = r_squared;
            neuron_matrix(d).VIS_fits_interp = fits_interpolated;
        else
            neuron_matrix(d).MULTI_RF_area = RF_area;
            neuron_matrix(d).MULTI_r_squared = r_squared;
            neuron_matrix(d).MULTI_fits_interp = fits_interpolated;
        end
    end

end

%% Let's save the centroid coordinates for all stimuli 

n_mod = 3;
recs = unique(extractfield(neuron_matrix,'Experiment_ID'));

for i = 1:length(recs)
    ix = recs(i);
    pos = find(extractfield(neuron_matrix,'Experiment_ID')==ix);

    for p = 1:length(pos)

        px = pos(p);

        for mod = 1:n_mod
            if mod == 1
                interp = neuron_matrix(px).AUD_fits_interp;
            elseif mod == 2
                interp = neuron_matrix(px).VIS_fits_interp;
            else
                interp = neuron_matrix(px).MULTI_fits_interp;
            end

            % Threshold to binary mask using half peak
            half_peak = (max(interp(:)) - min(interp(:))) / 1.3;
            half_peak = half_peak + min(interp(:));
            interp(interp < half_peak) = 0;
            interp(interp >= half_peak) = 1;

            % Find the coordinates of the nonzero pixels
            [row, col] = find(interp);

            % Calculate the centroid
            centroid_row = mean(row);
            centroid_col = mean(col);

            % Define the range of the original scale
            min_original = 1;
            max_x = size(interp,2);

            % Define the range of the target scale
            min_target = min(neuron_matrix(px).slices_degrees);
            max_target = max(neuron_matrix(px).slices_degrees);

            % Calculate the percentage of the value in the original range
            percentage = (centroid_col - min_original) / (max_x - min_original);

            % Interpolate the value in the target range
            col_deg = min_target + percentage * (max_target - min_target);

            % same for Y 
            max_y = size(interp,1);

            % Define the range of the target scale
            min_target = -36;
            max_target = 36;

            % Calculate the percentage of the value in the original range
            percentage = (centroid_row - min_original) / (max_y - min_original);

            % Interpolate the value in the target range
            row_deg = min_target + percentage * (max_target - min_target);

            coords = [col_deg,row_deg];

            if mod == 1
                neuron_matrix(px).AUD_coords = coords;
            elseif mod == 2
                neuron_matrix(px).VIS_coords = coords;
            else
                neuron_matrix(px).MULTI_coords = coords;
            end
        end
    end
end
disp('Done!')


%% This is a good moment to save everything

save(file_path,'neuron_matrix','-v7.3')

%% Save it as a dataset to upload in python.

% fits interp
retinotopy_dataset.VIS_fits_interp = reshape(extractfield(neuron_matrix,'VIS_fits_interp'),401,601,[]);
retinotopy_dataset.AUD_fits_interp = reshape(extractfield(neuron_matrix,'AUD_fits_interp'),401,601,[]);
retinotopy_dataset.MULTI_fits_interp = reshape(extractfield(neuron_matrix,'MULTI_fits_interp'),401,601,[]);

%peak FR
retinotopy_dataset.VIS_peaks = reshape(extractfield(neuron_matrix,'VIS_peaks'),35,[])';
retinotopy_dataset.AUD_peaks = reshape(extractfield(neuron_matrix,'AUD_peaks'),35,[])';
retinotopy_dataset.MULTI_peaks = reshape(extractfield(neuron_matrix,'MULTI_peaks'),35,[])';

retinotopy_dataset.VIS_r = extractfield(neuron_matrix,'VIS_r_squared')';
retinotopy_dataset.AUD_r = extractfield(neuron_matrix,'AUD_r_squared')';

% output modality 
retinotopy_dataset.VIS = extractfield(neuron_matrix,'VIS')';
retinotopy_dataset.AUD = extractfield(neuron_matrix,'AUD')';
retinotopy_dataset.AV = extractfield(neuron_matrix,'AV')';

% coords RF
retinotopy_dataset.VIS_coords = reshape(extractfield(neuron_matrix,'VIS_coords'),2,[])';
retinotopy_dataset.AUD_coords = reshape(extractfield(neuron_matrix,'AUD_coords'),2,[])';
retinotopy_dataset.MULTI_coords = reshape(extractfield(neuron_matrix,'MULTI_coords'),2,[])';

% animal ID and experiment ID
retinotopy_dataset.animal_ID= extractfield(neuron_matrix,'Animal_ID')';
retinotopy_dataset.experiment_ID = extractfield(neuron_matrix,'Experiment_ID')';
retinotopy_dataset.neuron_ID = extractfield(neuron_matrix,'Neuron_ID')';

retinotopy_dataset.n_col = 7;
retinotopy_dataset.n_rows = 5;
retinotopy_dataset.locs = n_col*n_rows;
retinotopy_dataset.stims = [1,locs; locs+1,locs*2; locs*2+1,locs*3];
retinotopy_dataset.slices_degrees = reshape(extractfield(neuron_matrix,'slices_degrees'),7,[])';

file_path = fullfile('Z:\home\shared\Gaia\Coliseum\Delays\paper_code\Datasets\neurons_datasets', ['retinotopy_dataset' '.mat']);
save(file_path,'retinotopy_dataset','-v7.3')
