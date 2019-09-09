import numpy as np
from PIL import Image
import hashlib

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
            return (self.mat, -2, False)
        
        # Update
        self.mat[action[0], action[1]] = 1

        # Check if User won
        if self._check_win(1):
            return (self.mat, 10, True)

        # Check for game end
        acts = self.action_space
        if len(acts) == 0:
            return (self.mat, 0, True)

        # If not done, then randomly spawn an 'O' on the board and recalculate the reward
        spawn_point = acts[np.random.choice(acts.shape[0])]
        self.mat[spawn_point[0], spawn_point[1]] = 2

        # Check if User lost
        if self._check_win(2):
            return (self.mat, -10, True)
        
        # If nothing wrong happens
        else:
            return (self.mat, -1, False)

    @property
    def state_hash(self):
        return hashlib.md5(str(self.mat).encode()).hexdigest()

    @staticmethod
    def box_coordinates(position):
        position = tuple(position)

        if position == (0, 0): return (60, 60, 180, 180)
        elif position == (0, 1): return (240, 60, 360, 180)
        elif position == (0, 2): return (420, 60, 540, 180)
        elif position == (1, 0): return (60, 240, 180, 360)
        elif position == (1, 1): return (240, 240, 360, 360)
        elif position == (1, 2): return (420, 240, 540, 360)
        elif position == (2, 0): return (60, 420, 180, 540)
        elif position == (2, 1): return (240, 420, 360, 540)
        elif position == (2, 2): return (420, 420, 540, 540)

    @staticmethod
    def _get_win_image_name(mat, num):
        locs = np.argwhere(mat==num).astype(np.int32)
        locs_has = lambda x: (locs == x).all(1).any()

        if (locs_has(np.array([0,0])) and locs_has(np.array([0,1])) and locs_has(np.array([0,2]))):
            return "hor_1"
        elif (locs_has(np.array([1,0])) and locs_has(np.array([1,1])) and locs_has(np.array([1,2]))):
            return "hor_2"
        elif (locs_has(np.array([2,0])) and locs_has(np.array([2,1])) and locs_has(np.array([2,2]))):
            return "hor_3"
        elif (locs_has(np.array([0,0])) and locs_has(np.array([1,0])) and locs_has(np.array([2,0]))):
            return "ver_1"
        elif (locs_has(np.array([0,1])) and locs_has(np.array([1,1])) and locs_has(np.array([2,1]))):
            return "ver_2"
        elif (locs_has(np.array([0,2])) and locs_has(np.array([1,2])) and locs_has(np.array([2,2]))):
            return "ver_3"
        elif (locs_has(np.array([0,0])) and locs_has(np.array([1,1])) and locs_has(np.array([2,2]))):
            return "dia_1"
        else:
            return "dia_2"

    @property
    def render_image(self):

        # Load the images
        board_pic = Image.open("assets/board.png")
        env_pic = Image.open("assets/o.png")
        user_pic = Image.open("assets/x.png")

        for i in range(self.mat.shape[0]):
            for j in range(self.mat.shape[1]):

                # If i,j th position is non empty
                if self.mat[i, j] != 0:
                    pos = self.box_coordinates([i,j])

                    if (self.mat[i, j] == 1):
                        board_pic.paste(user_pic, pos, user_pic)
                    else:
                        board_pic.paste(env_pic, pos, env_pic)
        
        # If anybody wins, draw a red line
        if self._check_win(1):
            red_line = Image.open("assets/" + self._get_win_image_name(self.mat, 1) + ".png")
            board_pic.paste(red_line, (0, 0, 600, 600), red_line)
            
        elif self._check_win(2):
            red_line = Image.open("assets/" + self._get_win_image_name(self.mat, 2) + ".png")
            board_pic.paste(red_line, (0, 0, 600, 600), red_line)
        
        return np.array(board_pic)
