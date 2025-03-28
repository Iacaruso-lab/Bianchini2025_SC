%% load the dataframe

clc
clear all
df = readtable('data_all_mean_25n.csv');

%% run model

% depending on what you are testing use one or the other formula
formula = ['accuracy ~ 1 + n_vis + n_aud + n_multi + n_delay2 + (1|animal_ID)'];
formula = ['accuracy ~ 1 + DSI + reliability + MII + abs_MII + (1|animal_ID)'];

model1 = fitlme(df, formula) %% area as a random effect
res0 = model1.Coefficients;
results = anova(model1)

%% extract the LMM fit to plot in python
names = ['Nvis','Naud','Nmulti','Ndelay'];

for i=2:height(res0)

    name = res0.Name{i}
    if contains(name, 'Intercept')
        name = 'Intercept';
    end
    estimate = res0.Estimate(i);
    lower = res0.Lower(i);
    upper = res0.Upper(i);
    pVal = res0.pValue(i);
    k = [estimate,pVal, lower, upper];
    eval(sprintf('res.%s = k;',name));%add the flattened variable onto a struct

    %get confidence interval
    if ~contains(name, 'Intercept')
        tblnew = table();
        for n=1:length(model1.PredictorNames)
            name0 = model1.PredictorNames{n};
            if strcmp(name, name0)
                tblnew.(name0) = linspace(min(df.(name)),max(df.(name)), 100)';
            else
                tblnew.(name0) = repmat(mean(df.(name0)), 100, 1);
            end

        end

        [ypred,yCI,DF] = predict(model1,tblnew);
        eval(sprintf('fitLines.%s = ypred;',name));%add the flattened variable onto a struct
        eval(sprintf('fitCI.%s = yCI;',name));%add the flattened variable onto a struct

        var = tblnew.(name);

        figure,
        scatter(df.(name), df.accuracy, 20,'filled','k');

        xlabel(name);
        ylabel('accuracy');
        pValStr = sprintf('%.3f', pVal);
        title(['Relationship between ', name, ' and MII, p-value: ', pValStr]);
        hold on;
        plot(tblnew.(name),ypred,'k')
        % Plot the confidence intervals
        x = tblnew.(name);
        ciLower = yCI(:, 1);
        ciUpper = yCI(:, 2);
        fill([x; flipud(x)], [ciLower; flipud(ciUpper)], 'k', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

    end

    filename = sprintf('accuracy_prediction_results_%s.mat', names{i-1}); % Using i-1 because MATLAB indexing starts at 1
    save(filename, 'ypred', 'yCI', 'var'); % Save the variables

end


