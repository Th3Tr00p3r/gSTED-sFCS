addpath(genpath('/Users/oleg/Documents/MATLAB/FCS Oject Oriented/'));

%%
load '/Users/oleg/Google Drive/10.5.18/AngularScan/120/AngularScan300bpConf2.mat'

%%
fileID = fopen('/Users/oleg/Documents/Python programming/FCS Python/testData.bin','w');
fwrite(fileID,FullData.Data)
fclose(fileID);


%% test result

Spython = load('/Users/oleg/Documents/Python programming/FCS Python/procData.mat');


%%
Smat = ConvertFPGAdataToPhotons_v3(FullData.Data)