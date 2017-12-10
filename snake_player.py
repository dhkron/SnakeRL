import os
import time
import random
import collections
import numpy as np
import tensorflow as tf

import snake_game

EXP_BUFFER_SIZE = 100000
EXP_BUFFER_BATCH_SIZE = 64
MAX_EPISODES = 50000
MAX_EPISODE_STEPS = 1000
DISCOUNT = 0.99
LEARNING_RATE = 1e-4

class ExpBuffer:
    def __init__(self):
        self.buffer_size = EXP_BUFFER_SIZE
        self.count = 0
        self.buffer = collections.deque()

    def add(self, s, a, r, s2):
        experience = (s, a, r, s2)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        '''     
        batch_size specifies the number of experiences to add 
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least 
        batch_size elements before beginning to sample from it.
        '''
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        s2_batch = np.array([_[3] for _ in batch])

        return s_batch, a_batch, r_batch, s2_batch

    def clear(self):
        self.deque.clear()
        self.count = 0

class Player:
    def __init__(
        self,
        w,
        h,
    ):
        self.w = w
        self.h = h
        self.exp_buffer = ExpBuffer()

        tf.reset_default_graph()

        self.x_board = tf.placeholder(tf.float32, shape=[None, h, w])

        x_board_reshaped = tf.reshape(self.x_board, [-1, h, w, 1])
        h_conv1 = tf.contrib.layers.conv2d(
            inputs=x_board_reshaped,
            num_outputs=32,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding='SAME',
        )
        h_conv1_flat = tf.reshape(h_conv1, [-1, w * h * 32])

        h_fc1 = tf.contrib.layers.fully_connected(
            inputs=h_conv1_flat,
            num_outputs=w * h * 32,
        )

        self.Q = tf.contrib.layers.fully_connected(
            inputs=h_fc1,
            num_outputs=3,
            activation_fn=None,
        )
        
        self.action = tf.argmax(self.Q, 1)

        self.reward_plus_discounted_next_max_Q = tf.placeholder(tf.float32, shape=[None, 3])
        loss = tf.reduce_sum(
            tf.square(
                self.reward_plus_discounted_next_max_Q - self.Q,
            ),
        )
        self.train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    def get_action(
        self,
        board_state,
        e=0.1,
    ):
        action = self.sess.run(
            self.action,
            feed_dict={
                self.x_board: [board_state],
            },
        )
        action = action[0]
        if np.random.rand(1) < e:
            action = random.choice([0, 1, 2])

        return action

    def train_batch(
        self,
        states,
        actions,
        rewards,
        next_states,
    ):
        q = self.sess.run(
            self.Q,
            feed_dict={
                self.x_board: states,
            },
        )
        next_q = self.sess.run(
            self.Q,
            feed_dict={
                self.x_board: next_states,
            },
        )
       
        # q is is [1,3], so in order to keep loss modify just q[action]
        for i, r in enumerate(rewards):
            max_next_q = np.max(next_q[i,:])
            q[i, actions[i]] = max_next_q * DISCOUNT + r
        
        self.sess.run(
            [self.train_step],
            feed_dict={
                self.x_board: states,
                self.reward_plus_discounted_next_max_Q: q, 
            },
        )
    
    def train_episode(
        self,
        game,
        episode_number,
    ):
        game.start()
        total_reward = 0
        j = 0

        for j in range(MAX_EPISODE_STEPS):
            current_state = game.get_board()
            current_action = self.get_action(
                board_state=current_state,
                e=1/(episode_number/100+10),
            )

            terminal, reward = game.step(current_action)
            next_state = game.get_board()
            total_reward += reward

            self.exp_buffer.add(
                current_state,
                current_action,
                reward,
                next_state,
            )

            if self.exp_buffer.size() > EXP_BUFFER_BATCH_SIZE:
                s_batch, a_batch, r_batch, s2_batch = self.exp_buffer.sample_batch(EXP_BUFFER_BATCH_SIZE)
                self.train_batch(
                    states=s_batch,
                    actions=a_batch,
                    rewards=r_batch,
                    next_states=s2_batch,
                )

            if terminal:
                break

        return total_reward

    def train(
        self,
    ):
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        game = snake_game.Game(self.w, self.h)
        total_episodes_reward = 0
        
        for i in range(MAX_EPISODES):
            total_episodes_reward += self.train_episode(
                game=game,
                episode_number=i,
            )
          
            if i%100 == 0:
                print(
                    'Total reward for episode {episode_number}: {reward}'.format(
                        episode_number=i,
                        reward=total_episodes_reward/100 ,
                    )
                )
                total_episodes_reward=0

    def play(
        self,
    ):
        while True:
            g = snake_game.Game(self.w, self.h)
            g.start()
            input('Waiting for you...')
            while True:
                try:
                    g.draw_with_sleep()
                    
                    action = self.get_action(
                        board_state=g.get_board(),
                        e=0,
                    )
                    
                    terminal, reward = g.step(action)
                    if terminal:
                        if reward:
                            g.draw_with_clear()
                        break
                except KeyboardInterrupt:
                    break

    @classmethod
    def weight_variable(cls, shape):
        initial = tf.truncated_normal(
            stddev=0.1,
            shape=shape,
        )
        return tf.Variable(initial)

    @classmethod
    def bias_variable(cls, shape):
        initial = tf.constant(
            value=0.1,
            shape=shape,
        )
        return tf.Variable(initial)

    @classmethod
    def conv2d(cls, x, W):
        return tf.nn.conv2d(
            input=x,
            filter=W,
            strides=[1, 1, 1, 1],
            padding='SAME',
        )

    @classmethod
    def max_pool_2x2(cls, x):
        return tf.nn.max_pool(
            x,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
        )
