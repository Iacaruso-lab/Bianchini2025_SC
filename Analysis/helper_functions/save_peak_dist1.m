function save_peak_dist1(output_boot,d,D)

% Define the base directory
baseDir = '/camp/lab/iacarusof/home/shared/Gaia/Coliseum/Delays/boot_all';

% Construct the directory path
dirPath = fullfile(baseDir, ['results_D' num2str(D)]);

% Format the file name using sprintf
fileName = sprintf('output-%d', d);

% Define the extension
extension = '.mat';

% Construct the full file path
matname = fullfile(dirPath, [fileName extension]);

save(matname,'output_boot')

end
