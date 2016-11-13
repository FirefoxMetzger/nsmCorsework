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
        function [num_steps, STM] = rndEpisode(this)
            gamma = 0.9;
            STM = zeros(20,3);
            num_steps = 1;

            this.world.reset();
            while ~this.world.isGoal()
                o = this.world.observe();
                a = randi(4);
                r = this.world.take_action(a);
                
                STM = circshift(STM,[-1 0]);
                STM(20,:) = [o a r];
                num_steps = num_steps + 1;
            end
            
            STM(:,3) = STM(20,3) * gamma.^(19:-1:0);
            
            if num_steps < 20
                STM(1:(20-num_steps),3) = 0;
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