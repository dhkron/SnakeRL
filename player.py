import os
import time
import random
import collections
import numpy as np
import tensorflow as tf

import game
import expbuffer

EXP_BUFFER_SIZE = 100000
EXP_BUFFER_BATCH_SIZE = 64
MAX_EPISODES = 100000
MAX_EPISODE_STEPS = 1000
DISCOUNT = 0.99
LEARNING_RATE = 1e-4
REPRESENTATION_CHANNELS = 3


class Player:
    def __init__(
        self,
        w,
        h,
    ):
        self.w = w
        self.h = h
        self.exp_buffer = expbuffer.ExpBuffer(
            size=EXP_BUFFER_SIZE
        )

        tf.reset_default_graph()

        self.x= tf.placeholder(tf.float32, shape=[None, h * w * REPRESENTATION_CHANNELS])

        x_reshaped = tf.reshape(self.x, [-1, h, w, REPRESENTATION_CHANNELS])
        h_conv1 = tf.contrib.layers.conv2d(
            inputs=x_reshaped,
            num_outputs=32,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding='SAME',
        )
        h_conv2 = tf.contrib.layers.conv2d(
            inputs=h_conv1,
            num_outputs=32,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding='SAME',
        )
        h_conv2_flat = tf.reshape(h_conv2, [-1, w * h * 32])

        h_fc1 = tf.contrib.layers.fully_connected(
            inputs=h_conv2_flat,
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
        state_rep,
        e=0.1,
    ):
        action = self.sess.run(
            self.action,
            feed_dict={
                self.x: [state_rep],
            },
        )
        action = action[0]
        if np.random.rand(1) < e:
            action = random.choice([0, 1, 2])

        return action

    def get_rep(
        self,
        board_states,
    ):
        rep = np.array(board_states).reshape(-1)
        return rep

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
                self.x: states,
            },
        )
        next_q = self.sess.run(
            self.Q,
            feed_dict={
                self.x: next_states,
            },
        )
       
        for i, r in enumerate(rewards):
            max_next_q = np.max(next_q[i,:])
            q[i, actions[i]] = max_next_q * DISCOUNT + r
        
        self.sess.run(
            [self.train_step],
            feed_dict={
                self.x: states,
                self.reward_plus_discounted_next_max_Q: q, 
            },
        )

    def get_new_board_history(
        self,
        g,
    ):
        board_history = collections.deque(
            maxlen=REPRESENTATION_CHANNELS, 
        )
        init_board_state = g.get_board().reshape(-1)
        for _ in range(REPRESENTATION_CHANNELS):
            board_history.append(
                init_board_state
            )

        return board_history

    
    def train_episode(
        self,
        g,
        episode_number,
    ):
        g.start()
        total_reward = 0
        j = 0

        board_history = self.get_new_board_history(
            g=g,
        )

        for j in range(MAX_EPISODE_STEPS):
            current_rep=self.get_rep(
                board_states=board_history,
            )
            current_action = self.get_action(
                state_rep=current_rep,
                e=1/(episode_number/100+10),
            )

            terminal, reward = g.step(current_action)
            next_state = g.get_board().reshape(-1)
            total_reward += reward

            board_history.append(
                next_state
            )

            next_rep=self.get_rep(
                board_states=board_history,
            )

            self.exp_buffer.add(
                current_rep,
                current_action,
                reward,
                next_rep,
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
        g = game.Game(self.w, self.h)
        total_episodes_reward = 0
        
        for i in range(MAX_EPISODES):
            total_episodes_reward += self.train_episode(
                g=g,
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
            g = game.Game(self.w, self.h)
            g.start()
            board_history = self.get_new_board_history(
                g=g,
            )
            input('Waiting for you...')
            while True:
                try:
                    g.draw_with_sleep()
                    
                    action = self.get_action(
                        state_rep=self.get_rep(
                            board_states=board_history,
                        ),
                        e=0,
                    )
                    
                    terminal, reward = g.step(action)
                    if terminal:
                        if reward:
                            g.draw_with_clear()
                        break

                    board_history.append(
                        g.get_board().reshape(-1)
                    )
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
