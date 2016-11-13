classdef random_agent < handle
    % An agent that interacts with the partially observable version of
    % McCallum's grid world.
    % The agent will take a random action in each state.
    %
    % (c) 2016 Sebastian Wallkötter
    
    methods
        function [num_steps, ret_mat] = rndEpisode(~)
            gamma = 0.9;
            
            episode = zeros(1,3);
            pos = 1;
            
            world = simulator();
            world.reset();
            while ~world.isGoal()
                o = world.observe();
                a = randi(4);
                r = world.take_action(a);
                
                episode(pos,:) = [o a r];
                pos = pos + 1;
            end
            
            num_steps = size(episode,1);
            
            for idx = (num_steps-1):-1:1
                episode(idx,3) = episode(idx,3) + gamma * episode(idx+1,3);
            end
            
            if num_steps >= 20
                ret_mat = episode((end-19):end,:);
            else
                ret_mat = zeros(20,3);
                first_idx = 21-num_steps;
                ret_mat(first_idx:end,:) = episode;
            end
        end
        function [steps, LTM] =  Trials(this,num_trials)
            LTM = zeros(20,3,num_trials);
            steps = zeros(1,num_trials);
            for idx = 1:num_trials
                [step,foo] = this.rndEpisode();
                steps(idx) = step;
                LTM(:,:,idx) = foo; 
            end
        end
    end
end