# SnakeRL

Run: `python3 snake.py`

## Some words about snake
This script was meant to defeat snake by only seeing the pixels on the board.
I used a simple DQN at first, but it was very slow, so I added an experience replay buffer.
This improve the results and learning rate greatly, but still never got to perfect play.

So I tried several optimizations as giving several frames of game each time. This failed miserably, I guess because it confused the heck out of the net.
Then, I used a priority exp buffer which did results pretty compareable to irignal DQN+Exp. I tried several types of priority queues but to no good.
Finally, I tried giving the DQN the previous action played, converting it to RNN of some sort. The previous action was added right after the conv layer. It gave comparable results, perhaps slightly better.

As a last attempt, I tried setting different value to different snake scales (i.e. head=2, ranging to tail=1) but it yielded worse results.


If running in ipython while editing code, should start with -
```
import sys

sys.dont_write_bytecode = True

%load_ext autoreload
%autoreload 2
```

### Bibliography

- [DeepMind Atari paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- [Deep Deterministic Policy Gradients in TensorFlow](http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html)
- [Simple Reinforcement Learning with Tensorflow Part 4: Deep Q-Networks and Beyond](https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb)
