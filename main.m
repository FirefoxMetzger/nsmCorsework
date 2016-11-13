% The main entry point to the RL assignment. Part of AINT511 at Plymouth
% University.
% 
% This function plots a graph displaying the results of num_trials many
% episodes with random action selection and num_trials many episodes using
% NSM for action selection.
% 
% (c) 2016 Sebastian Wallk�tter

rndAgent = random_agent();
nsmAgent = nsm_agent();
num_trials = 100;
[steps_rnd,~] = rndAgent.Trials(num_trials);
[steps_learn, ~] = nsmAgent.Trials(num_trials);

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