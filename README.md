# SnakeRL

Run: `python3 snake.py`

## TL;DR
These bunch of scripts were used by me to play around with reinforcement learning, given the task of defeating snake.
I tried many methods, focusing on 4x4 board size. Got moderate results, probabliy because of falling into local maxima or because snake states can not easily be generalized by a DQN.

## Some more words
This is an educational script, meant to defeat snake by only seeing the pixels on the board.
Therefore it contains many commits of attempts I did to figure it out.
At first I used a simple convolutional DQN, but it was very slow, so I added an experience replay buffer.
This greatly improved the results and learning rate, but still never got to perfect play.

So I tried several optimizations as giving several frames of game each time. This failed miserably, I guess because it confused the heck out of the net.
Then, I used a priority exp buffer which gave out pretty compareable results to orignal DQN+Exp. I tried several types of priority queues but to no good.
Finally, I tried giving the DQN the previous action played, converting it to RNN of some sort. The previous action was added right after the conv layer. It gave comparable results, perhaps slightly better.

As a last attempt, I tried setting different value to different snake scales (i.e. head=2, ranging to tail=1) but it yielded worse results.

A final model of 3 hidden layers of 128 neurons also gives nice results, but never satisfying.

## Some comments

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
