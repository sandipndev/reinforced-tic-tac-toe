import numpy as np

class TicTacToeEnv():
    """This is the environment on which the Tic Tac Toe game will run
    Convention:
        0 -> Empty
        1 -> X (User)
        2 -> O (Environment)
    Step funtion steps onto a given position and returns"""
    
    def reset(self):
        """Resets an episode and returns base environment observation"""
        self.mat = np.zeros(9).reshape(3,3).astype(np.int32)
        return self.mat

    @property
    def action_space(self):
        """Possible locations where User can hit"""
        return np.argwhere(self.mat==0).astype(np.int32)
    
    def _check_win(self, num):
        """Check Wins"""
        locs = np.argwhere(self.mat==num).astype(np.int32)

        # To check if an array is in this
        locs_has = lambda x: (locs == x).all(1).any()

        return  (locs_has(np.array([0,0])) and locs_has(np.array([0,1])) and locs_has(np.array([0,2]))) or \
                (locs_has(np.array([1,0])) and locs_has(np.array([1,1])) and locs_has(np.array([1,2]))) or \
                (locs_has(np.array([2,0])) and locs_has(np.array([2,1])) and locs_has(np.array([2,2]))) or \
                (locs_has(np.array([0,0])) and locs_has(np.array([1,0])) and locs_has(np.array([2,0]))) or \
                (locs_has(np.array([0,1])) and locs_has(np.array([1,1])) and locs_has(np.array([2,1]))) or \
                (locs_has(np.array([0,2])) and locs_has(np.array([1,2])) and locs_has(np.array([2,2]))) or \
                (locs_has(np.array([0,0])) and locs_has(np.array([1,1])) and locs_has(np.array([2,2]))) or \
                (locs_has(np.array([0,2])) and locs_has(np.array([1,1])) and locs_has(np.array([2,0])))

    def step(self, action):
        """Steps into another state in the current episode"""
    
        # Check if the given position is empty
        if self.mat[action[0], action[1]] != 0:
            return (self.mat, 0, None)
        
        # Update
        self.mat[action[0], action[1]] = 1

        # Check if User won
        if self._check_win(1):
            return (self.mat, 1, True)

        # Check for game end
        acts = self.action_space
        if len(acts) == 0:
            return (self.mat, 0, True)

        # If not done, then randomly spawn an 'O' on the board and recalculate the reward
        spawn_point = acts[np.random.choice(acts.shape[0])]
        self.mat[spawn_point[0], spawn_point[1]] = 2

        # Check if User lost
        if self._check_win(2):
            return (self.mat, -0.5, True)
        
        # If nothing wrong happens
        else:
            return (self.mat, 0, False)

    def render(self):
        pass