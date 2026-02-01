clear;
clc;

% ------------ INPUTS -------------------
load('all_behav.mat')
load('all_mats.mat')

no_sub = size(all_mats,3);
true_prediction_r =network_CPM_function(all_mats, all_behav, 0.05);

% permutation
no_iterations     = 1000;
prediction_r      = zeros(no_iterations, 1);
prediction_r(1,1) = true_prediction_r;

% via random shuffles of data labels
for it = 2:no_iterations
      fprintf('Permutation iteration %d of %d\n', it, no_iterations);
      new_behav = all_behav(randperm(no_sub));
      r=network_CPM_function(all_mats, new_behav, 0.05);
      prediction_r(it,1) = r;
end

sorted_prediction_r = sort(prediction_r(:,1), 'descend');
position_r          = find(sorted_prediction_r==true_prediction_r);
pval_r              =position_r(1) / no_iterations;


