    % The main entry point to the RL assignment. Part of AINT511 at Plymouth
% University.
% 
% This function plots a graph displaying the results of num_trials many
% episodes with random action selection and num_trials many episodes using
% NSM for action selection.
% 
% (c) 2016 Sebastian Wallkoetter

% change as desired
num_trials = 100;

% gather data -- may take time
rndAgent = random_agent();
[steps_rnd,~] = rndAgent.Trials(num_trials);

nsmAgent = nsm_agent();
[steps_learn, ~] = nsmAgent.Trials(num_trials);

% plot nice output graphs
subplot(1,2,1);
plot(steps_rnd);
ylim([0 500]);
title('Random Trials');
ylabel('Number of Steps');
xlabel('Episodes');

subplot(1,2,2);
plot(steps_learn);
ylim([0 500]);
title('NSM Trials');
ylabel('Number of Steps');
xlabel('Episodes');