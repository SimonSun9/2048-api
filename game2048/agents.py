import numpy as np


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction
    
class MyAgent(Agent):

    def __init__(self, game, display=None):
        super().__init__(game, display)
        self.model_256 = net.RNN()
        self.model_512 = net.RNN()
        self.model_1024 = net.RNN()
        self.model_256.load_state_dict(torch.load('model/model128.pkl', map_location=torch.device('cpu')))
        self.model_512.load_state_dict(torch.load('model/model512.pkl', map_location=torch.device('cpu')))
        self.model_1024.load_state_dict(torch.load('model/model1024.pkl', map_location=torch.device('cpu')))

    def step(self):
        board1 = self.game.board
        board1[board1 == 0] = 1
        board1 = np.log2(board1)
        board2 = np.transpose(board1)
        input = np.vstack((board1, board2)).reshape((1, 8, 4))
        input = torch.from_numpy(input)
        if np.max(board1) < 8:
            output = self.model_256(input.float())
        elif np.max(board1) < 9:
            output = self.model_512(input.float())
        else:
            output = self.model_1024(input.float())
        direction = torch.max(output, 1)[1]
        return direction

class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction
