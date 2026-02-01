clear;
clc;

% ------------ INPUTS -------------------
load('all_mats.mat')
load('all_behav.mat')
load('network_label.mat')

roi_network = network_label(:,2); 
nNetwork = length(unique(roi_network));
mask = triu(true(nNetwork),1);

% threshold for feature selection
thresh_all=[0.05,0.01,0.005,0.001,0.0005];
no_thresh=numel(thresh_all);
no_sub = size(all_mats,3);
no_node = size(all_mats,1);
R_all = zeros(no_thresh,1);
P_all = zeros(no_thresh,1);

for aa=1:no_thresh
    
    thresh = thresh_all(aa);
    behav_pred = zeros(no_sub,1);
    for leftout = 1:no_sub
        fprintf('\n Leaving out subj # %6.3f',leftout);
        
        % leave out subject from matrices and behavior
        train_mats = all_mats;
        train_mats(:,:,leftout) = [];
        
        train_behav = all_behav;
        train_behav(leftout) = [];

        network_conn = zeros(nNetwork, nNetwork, no_sub-1);
        for ss=1:no_sub-1
           mat = train_mats(:,:,ss);
            for i = 1:nNetwork
                for j = 1:nNetwork
                    roi_i = find(roi_network == i);
                    roi_j = find(roi_network == j);
                    submat = mat(roi_i, roi_j);
                    network_conn(i,j,ss) = sum(submat(:));
                end
            end
        end
        network_features = zeros(sum(mask(:)), no_sub-1);
        for ss = 1:no_sub-1
            temp = network_conn(:,:,ss);
            network_features(:,ss) = temp(mask);
        end
        network_features=network_features';
        [r_mat,p_mat]=corr(train_behav,network_features,'type','Spearman');
        features_idx=find(p_mat<thresh);
        features_final=network_features(:,features_idx);
      
        % build model on TRAIN subs
        b = regress(train_behav, [features_final, ones(no_sub-1,1)]);
        
        % run model on TEST sub
        test_mat = all_mats(:,:,leftout);
        network_conn_test=zeros(nNetwork,nNetwork);
        for i = 1:nNetwork
            for j = 1:nNetwork
                roi_i = find(roi_network == i);
                roi_j = find(roi_network == j);
                submat_test = test_mat(roi_i, roi_j);
                network_conn_test(i,j) = sum(submat_test(:));
            end
        end
        network_features_test=network_conn_test(mask);
        network_features_test=network_features_test';
        features_final_test=network_features_test(features_idx);
        features_final_test=[features_final_test,1];
        behav_pred(leftout) = dot(b', features_final_test);
    end
    
    % compare predicted and observed scores
    [R, P] = corr(all_behav,behav_pred)
    R_all (aa,1)=R;
    P_all (aa,1)=P;
end
