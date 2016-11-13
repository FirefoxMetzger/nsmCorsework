classdef nsm_agent < handle
    % An agent that interacts with the partially observable version of
    % McCallum's grid world.
    % The agent keeps a history of all past episodes (max. 20 steps).
    % This information is used to make decisions about the current action,
    % by comparing chains of previous observations and actions.
    %
    % (c) 2016 Sebastian Wallkötter
    properties(Access=private)
        episodes_stored = 0;
        steps_stored = 0;
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
                
                this.steps_stored = min(20,this.steps_stored + 1);
                this.STM = circshift(this.STM,[-1 0]);
                this.STM(20,:) = [o a r];
                num_steps = num_steps + 1;
            end
            
            this.STM(:,3) = this.STM(20,3) * gamma.^(19:-1:0);
            this.STM(1:(20-this.episodes_stored),3) = 0;
            
           ret_mat = this.STM;
        end
        function a = NSMSelectAction(this, o)
            if rand < 0.1
                a = randi(4);
            else
                kNN = this.kNearest(o);
                rewards = zeros(1,4);
                for idx = 1:4
                    same_action = (kNN(:,2) == idx);
                    rewards(idx) = mean(kNN(same_action,3));
                end
                rewards(isnan(rewards)) = 0;
                [~,a] = max(rewards);
            end
        end
        function score = proximity(this, episode, step, o)
            % define some nice names
            LTM = this.LTM(:,:,1:(this.episodes_stored));       %#ok<PROPLC>
            past = LTM(1:step-1,1:2,episode);                   %#ok<PROPLC>
            present = LTM(step,1:2,episode);                    %#ok<PROPLC>
            
            % if observation differs from LTM's present observation,
            % score is 0
            if present(1) ~= o
                score = 0;
                return;
            end
            
            % compare LTM's past to LTM past to see how close it is
            STM_size = this.steps_stored;
            past_size = size(past, 1);
            
            %exctract the last 'common' elements from both elements
            common = (min(STM_size,past_size)-1):-1:0;
            
            idx_past = past_size - common;
            past = past(idx_past,:);
            
            idx_STM = STM_size - common;
            STM = this.STM(idx_STM,1:2);                        %#ok<PROPLC>
            
            %find out how many (o,a) pairs match
            matches = all(past == STM,2);                       %#ok<PROPLC>
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