classdef nsm_agent < handle
    % An agent that interacts with the partially observable version of
    % McCallum's grid world.
    % The agent keeps a history of all past episodes (max. 20 steps).
    % This information is used to make decisions about the current action,
    % by comparing chains of previous observations and actions.
    %
    % (c) 2016 Sebastian Wallk�tter
    
    properties(Access=private)
        episodes_stored = 0;
        world = simulator();
        STM = zeros(20,3);
        LTM = zeros(20,3,0);
    end
    
    methods
        function [steps, LTM] =  Trials(this,num_trials)
            this.LTM = zeros(20,3,num_trials);
            
            steps = zeros(1,num_trials);
            for idx = 1:num_trials
                [step,foo] = this.NSMEpisode();
                steps(idx) = step;
                this.LTM(:,:,idx) = foo;
                this.episodes_stored = this.episodes_stored + 1;
            end
            LTM = this.LTM;
        end

    end
    methods(Access = private)
        function [num_steps, ret_mat] = NSMEpisode(this)
            gamma = 0.9;
            num_steps = 0;
            
            this.world.reset();
            this.STM = zeros(20,3);
            while ~this.world.isGoal()
                o = this.world.observe();
                a = this.NSMSelectAction(o);
                r = this.world.take_action(a);
                
                this.STM = circshift(this.STM,[-1 0]);
                this.STM(20,:) = [o a r];
                num_steps = num_steps + 1;
            end
            
            this.STM(:,3) = this.STM(20,3) * gamma.^(19:-1:0);
            this.STM(1:(20-this.episodes_stored),3) = 0;
            
           ret_mat = this.STM;
        end
        function a = NSMSelectAction(this, o)
            % selects an action based on actions and observations 
            % (STM) made in this episode and past experiences (LTM).
            % The action selection is epsilon-greedy, with epsilon = 10 %.
            % Input: o -- the current observation
            % Output: a -- action chosen
            
            if rand < 0.1
                % pick a random action
                a = randi(4);
                
            else
                % select an action based on past experiences
                
                % find the 10 closest POMDP states observed in the past and
                % average their discounted reward with respect to the first
                % action taken.
                kNN = this.kNearest(o);
                rewards = zeros(1,4);
                for idx = 1:4
                    same_action = (kNN(:,2) == idx);
                    rewards(idx) = mean(kNN(same_action,3));
                end
                rewards(isnan(rewards)) = 0;
                
                % select the best action. If tie for multiple, pick random
                % witin those.
                max_val = max(rewards);
                available_actions = 1:4;
                available_actions = ...
                    (available_actions(abs(rewards-max_val) < 0.01));
                
                if size(available_actions, 2) == 1
                    a = available_actions;
                else
                    a = randsample(available_actions,1);
                end
            end
        end
        function score = proximity(this, episode, step, o)
            % define some nice names
            present = this.LTM(step,1:2,episode);       
            
            % if observation differs from LTM's present observation,
            % score is 0
            if present(1) ~= o
                score = 0;
                return;
            end
            
            past = this.LTM(1:step-1,1:2,episode);
            zero_rows = all(past == 0,2);
            past(zero_rows,:) = [];
            
            local_STM = this.STM(:,1:2);
            zero_rows = all(local_STM == 0,2);
            local_STM(zero_rows,:) = [];
            
            
            % compare LTM's past to STM past to see how close it is
            STM_size = size(local_STM,1);
            past_size = size(past, 1);
            
            %exctract the last 'common' elements from both elements
            common = (min(STM_size,past_size)-1):-1:0;
            
            past = past(end - common,:);
            local_STM = this.STM(end - common,1:2);
            
            %find out how many (o,a) pairs match
            matches = all(past == local_STM,2);
            num_matches = find([matches; 0]==0, 1);
            score = 1 + (num_matches-1);
        end
        function kNN = kNearest(this, o)
            kNN = zeros(10,4);
            for step = 20:-1:2
                for episode = 1:this.episodes_stored
                    score = this.proximity(episode,step,o);
                    if kNN(1,4) <= score && score > 0
                        kNN(1,:) = [this.LTM(step,:,episode) score];
                        
                        [~,idx] = sort(kNN(:,4));
                        kNN = kNN(idx,:);
                    end
                end
            end
        end
    end
end