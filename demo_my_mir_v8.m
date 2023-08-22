clear;clc;
load mir5k_v2.mat;
fprintf('MIR Flickr_cnn dataset loaded...\n');
%% parameter settings
run = 1;
map = zeros(run, 1);
bits = [16 32 64 128];
alpha = 1e1; 
beta = 1e5; 
delta = 1e-5;  
alm = 1e3; 
alm2 = 1e5; 
gamma = 1e-5;
theta = 1e-3;
results = zeros(1, 11);
%%
for bi = 4
param.bits=bits(bi);
S = param.bits*S;
S(S==0) = -param.bits;   

for i = 1 : run
%fprintf('run %d starts...\n', i);
%% centralization
%fprintf('centralizing data...\n');
Ntrain = size(I_tr,1);
n_anchors = 500;
sample = randsample(Ntrain, n_anchors);
anchorI = I_tr(sample,:);
anchorT = T_tr(sample,:);
sigmaI = 50;
sigmaT = 130;
PhiI = exp(-sqdist(I_tr,anchorI)/(2*sigmaI*sigmaI));
PhiI = [PhiI, ones(Ntrain,1)];
PhiT = exp(-sqdist(T_tr,anchorT)/(2*sigmaT*sigmaT));
PhiT = [PhiT, ones(Ntrain,1)];
Phi_testI = exp(-sqdist(I_te,anchorI)/(2*sigmaI*sigmaI));
Phi_testI = [Phi_testI, ones(size(Phi_testI,1),1)];
Phi_testT = exp(-sqdist(T_te,anchorT)/(2*sigmaT*sigmaT));
Phi_testT = [Phi_testT, ones(size(Phi_testT,1),1)];
Phi_dbI = exp(-sqdist(I_db,anchorI)/(2*sigmaI*sigmaI));
Phi_dbI = [Phi_dbI, ones(size(Phi_dbI,1),1)];
Phi_dbT = exp(-sqdist(T_db,anchorT)/(2*sigmaT*sigmaT));
Phi_dbT = [Phi_dbT, ones(size(Phi_dbT,1),1)];
PhiI = PhiI';
PhiT = PhiT';
Phi_testI = Phi_testI';
Phi_testT = Phi_testT';
Phi_dbI = Phi_dbI';
Phi_dbT = Phi_dbT';
Y = L_tr';
param.bits = bits(bi); 
param.alpha = alpha;
param.beta = beta;
param.delta = delta;
param.alm = alm;
param.alm2 = alm2;
param.gamma = gamma;
param.theta = theta;
%% solve objective function
fprintf('start training...\n');
[iter_fin, lastF, B_tr, B_test, B_db] = solveMVLH(I_tr', T_tr', anchorI, anchorT, PhiI, PhiT, Phi_testI, Phi_testT, Phi_dbI, Phi_dbT, Y, S, param);
%% calculate hash codes
Btrain = compactbit(B_tr>0);
Bbase = compactbit(B_db>0);
Bquery = compactbit(B_test>0);
%% evaluate
tic
fprintf('start evaluating...\n');
Dhamm = hammingDist(Bbase+2, Bquery+2);
[P] = perf_metric4Label( L_db, L_te, Dhamm);
map(i) = P;
fprintf('Query time:\n');
toc
% my_topN_mir128 = TOPK(Dhamm,L_db,L_te,param.bits);
% save('my_topN_mir128.mat', 'my_topN_mir128');
end
Rt = [iter_fin, param.alpha, param.beta, param.delta, param.alm, param.alm2, param.gamma, param.theta, sigmaI, sigmaT, mean(map)];
results = [results; Rt];
% save('results128.mat', 'results');
fprintf('====== bits=%d, mAP=%.4f ======\n\n', param.bits, mean(map));
end

