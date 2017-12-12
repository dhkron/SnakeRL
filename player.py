import os
import time
import random
import collections
import numpy as np
import tensorflow as tf

import game
import expbuffer
import history

EXP_BUFFER_SIZE = 1000000
EXP_BUFFER_BATCH_SIZE = 64
MAX_EPISODES = 100000
MAX_EPISODE_STEPS = 1000
DISCOUNT = 0.99
LEARNING_RATE = 1e-4
REPRESENTATION_CHANNELS = 1


class Player:
    def __init__(
        self,
        w,
        h,
    ):
        self.w = w
        self.h = h
        self.exp_buffer = expbuffer.ExpBuffer(
            size=EXP_BUFFER_SIZE,
        )

        tf.reset_default_graph()

        self.x= tf.placeholder(tf.float32, shape=[None, h,  w,  REPRESENTATION_CHANNELS])

        #x_reshaped = tf.reshape(self.x, [-1, h, w, REPRESENTATION_CHANNELS])
        h_conv1 = tf.contrib.layers.conv2d(
            inputs=x,
            num_outputs=1,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding='SAME',
        )
        h_conv2 = tf.contrib.layers.conv2d(
            inputs=h_conv1,
            num_outputs=32,
            kernel_size=[4, 4],
            stride=[1, 1],
            padding='SAME',
        )
        h_conv_flat = tf.layers.flatten(
            inputs=h_conv2,
        )
        h_fc1 = tf.contrib.layers.fully_connected(
            inputs=h_conv_flat,
            num_outputs=1024,
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

    def train_episode(
        self,
        g,
        episode_number,
    ):
        total_reward = 0
        j = 0
        
        initial_state = g.reset()
        
        board_history = history.History(
            num_channels=REPRESENTATION_CHANNELS,
        )
        board_history_add(initial_state)

        for j in range(MAX_EPISODE_STEPS):
            current_rep=board_history.get_rep()
            current_action = self.get_action(
                state_rep=current_rep,
                e=.05+.95/(1+episode_number/1000),
            )

            next_state, terminal, reward = g.step(current_action)
            total_reward += reward

            board_history.add(next_state)
            next_rep = board_history.get_rep()

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

            initial_state = g.reset()
            
            board_history = history.History(
                num_channels=REPRESENTATION_CHANNELS,
            )
            board_history_add(initial_state)
            
            input('Waiting for you...')
            while True:
                try:
                    g.draw_with_sleep()
                    
                    action = self.get_action(
                        state_rep=board_history.get_rep(),
                        e=0,
                    )

                    next_state, terminal, reward = g.step(action)
                    board_history.append(next_state)

                    if terminal:
                        if reward:
                            g.draw_with_clear()
                        break

                except KeyboardInterrupt:
                    break
