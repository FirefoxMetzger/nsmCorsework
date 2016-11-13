classdef simulator < handle
    % A small simulation of a variation of McCallum's grid-world.
    % Initialized in a random state. The world is only partially
    % observable.
    %
    % reset()        -- reset the world to a random starting state
    % observe()      -- get the observation of the current goal state
    % take_action(a) -- execute action a, returns the reward associated
    %                   with taking this action in the current state
    % isGoal(a)      -- check if in the goal state
    %
    % (c) 2016 Sebastian Wallkötter
    
    properties(Access=private)
        s = randi(11); % current state
        T = ones(4,11); % transition matrix
        observations = ones(11,1);
        reward_function
    end
    
    methods
        function obj = simulator(~)
            
            obj.T = [
                4 2   6   7   9   11  7   8   9   10  11
                1 2   3   4   5   6   8   9   10  11  11
                1 2   3   1   2   3   4   8   5   10  11
                1 2   3   4   5   6   7   7   8   9   10
                ];
            obj.observations = [14 14 14 10 10 10 9 5 1 5 3];
        end
        
        function r =  take_action(this, a)
            r = this.reward(a);
            this.s = this.T(a,this.s);
            next_state = this.s;
        end
        function o = observe(this)
            o = this.observations(this.s);
        end
        function reset(this)
            possible_states = 1:11;
            possible_states(2) = [];
            this.s = randsample(possible_states,1);
            state = this.s;
        end
        function b = isGoal(this)
            b = (this.s == 2);
        end
        
    end
    methods(Access = private)
        function r = reward(this, a)
            if this.s == 5 && a == 3
                r = 10;
            else
                r = 0;
            end
        end
    end
end