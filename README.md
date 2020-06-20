# 2048-api
用模仿学习，RNN神经网络实现2048游戏

# 代码结构
* [`game2048/`](game2048/): the main package.
    * [`game.py`](game2048/game.py): the core 2048 `Game` class.
    * [`agents.py`](game2048/agents.py): 包含 `Agent` 类。`MyAgent` 是它的一个子类 
    * [`displays.py`](game2048/displays.py): the `Display` class with instances, to show the `Game` state.
    * [`expectimax/`](game2048/expectimax): a powerful ExpectiMax agent by [here](https://github.com/nneonneo/2048-ai).
* [`explore.ipynb`](explore.ipynb): introduce how to use the `Agent`, `Display` and `Game`.
* [`static/`](static/): frontend assets (based on Vue.js) for web app.
* [`webapp.py`](webapp.py): run the web app (backend) demo.
* [`evaluate.py`](evaluate.py): evaluate your self-defined agent.
* [`net.py`](net.py): RNN 网络结构
* [`getdata.py`](getdata.py): 用来生成训练数据的程序，可以选择专家agent，也可以选择自己的agent。可以设置想要保存的分数段。
* [`train.py`](train.py): 用来训练模型的代码。训练出的模型保存在model_train文件夹中，训练数据在data文件夹中

# Requirements
* code only tested on linux system (ubuntu 16.04)
* Python 3 (Anaconda 3.6.3 specifically) with numpy and flask
* Pytorch

# 我的agent模型
```python
from game2048.agents import Agent

class MyAgent(Agent):

    def __init__(self, game, display=None):
        super().__init__(game, display)
        self.model_256 = net.RNN()
        self.model_512 = net.RNN()
        self.model_1024 = net.RNN()
        self.model_256.load_state_dict(torch.load('model128.pkl', map_location=torch.device('cpu')))
        self.model_512.load_state_dict(torch.load('model512.pkl', map_location=torch.device('cpu')))
        self.model_1024.load_state_dict(torch.load('model1024.pkl', map_location=torch.device('cpu')))

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

```
# 如何用MyAgent进行游戏


# Train my agent

# To compile the pre-defined ExpectiMax agent

```bash
cd game2048/expectimax
bash configure
make
```

# To run the web app
```bash
python webapp.py
```
![demo](preview2048.gif)

# LICENSE
The code is under Apache-2.0 License.

# For EE369 / EE228 students from SJTU
Please read course project [requirements](EE369.md) and [description](https://docs.qq.com/slide/DS05hVGVFY1BuRVp5). 
