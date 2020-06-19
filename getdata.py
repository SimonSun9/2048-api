from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
from game2048.agents import Agent, RandomAgent, ExpectiMaxAgent, MyAgent
import numpy as np
import pandas as pd


num = 10000
i = 0
results = []
direction = []
while i < num:
    game = Game(4, score_to_win=2048, random=False)
    agent_exp = ExpectiMaxAgent(game)
    agent = MyAgent(game)
    while (game.score <= 1024) and (not game.end):
        A = game.board
        A[A == 0] = 1
        A = np.log2(A)
        A = np.int32(A)
        A = A.reshape(16)
        dir = agent.step()
        # you can change the condition to get different data
        if game.score >= 512:
            dir_exp = agent_exp.step()
            results.append(A)
            direction.append(dir_exp)
        game.move(dir)
    if 0 == i % 100:
        # save the result every 100 games
        results = np.array(results)
        direction = np.array(direction)
        final_results = np.c_[results, direction]
        final_results = pd.DataFrame(final_results)
        final_results.to_csv("data/data_online_1024.csv", index=False, header=False, mode='a+')

        results = []
        direction = []
    i += 1




