classdef random_agent < handle
    % An agent that interacts with the partially observable version of
    % McCallum's grid world.
    % The agent will take a random action in each state.
    %
    % (c) 2016 Sebastian Wallkötter
    
    properties(Access=private)
        world = simulator();
    end
    
    methods
        function [steps, LTM] =  Trials(this,num_trials)
            % gather num_trials many episodes using random action
            % selection. For each Episode reccord the total number of steps
            % and the last 20 visited nodes.
            % Input:    num_trials  --- the number of trials to be run
            % Output:   steps       --- 1D array where steps(idx) corresponds to
            %                           the number of steps needed to reach the
            %                           goal in episode idx.
            %           LTM         --- A reccord of the last 20 reccorded
            %                           observations, actions and rewards.
            %                           LTM(20,:) is the last observed state.
            %                           If an episode requires less than 20
            %                           steps, the top rows are zero padded
            %                           until size(LTM) = [20 3];
            
            LTM = zeros(20,3,num_trials);
            steps = zeros(1,num_trials);
            for idx = 1:num_trials
                [step,foo] = this.rndEpisode();
                steps(idx) = step;
                LTM(:,:,idx) = foo;
            end
        end
    end
    
    methods(Access=private)
        function [num_steps, STM] = rndEpisode(this)
            % Generates an episode picking random actions in each state
            % Output:   num_steps   --- The total number of steps until the
            %                           goal was reached.
            %           ret_mat     --- The last 20 observations, actions
            %                           and rewards, stored in LTM format.
            
            gamma = 0.9;
            STM = zeros(20,3);
            num_steps = 0;
            
            % reset the world
            this.world.reset();
            
            % take a random action until something cool happens
            while ~this.world.isGoal()
                o = this.world.observe();
                a = randi(4);
                r = this.world.take_action(a);
                
                STM = circshift(STM,[-1 0]);
                STM(20,:) = [o a r];
                num_steps = num_steps + 1;
            end
            
            % calculate the discounted reward
            STM(:,3) = STM(20,3) * gamma.^(19:-1:0);
            STM(1:(20-num_steps),3) = 0;
            
        end
    end
end