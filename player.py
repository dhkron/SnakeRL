import os
import time
import random
import collections
import numpy as np
import tensorflow as tf

import game
import expbuffer
import history

EXP_BUFFER_SIZE = 100000
EXP_BUFFER_BATCH_SIZE = 64
MAX_EPISODES = 100000
MAX_EPISODE_STEPS = 1000
DISCOUNT = 0.99
LEARNING_RATE = 1e-4
#HISTORY_SIZE = 1


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

        self.prev_a = tf.placeholder(tf.float32, shape=[None, ])
        prev_a = tf.reshape(self.prev_a, shape=[-1, 1])

        self.x = tf.placeholder(tf.float32, shape=[None, h,  w])
        x_reshaped = tf.reshape(
            self.x,
            shape=[-1, h, w, 1],
        )

        h_conv1 = tf.contrib.layers.conv2d(
            inputs=x_reshaped,
            num_outputs=32,
            kernel_size=[4, 4],
            stride=[1, 1],
            padding='SAME',
        )
        h_conv_flat = tf.layers.flatten(
            inputs=h_conv1,
        )

        h_joined = tf.concat(
            [h_conv_flat, prev_a],
            1,
        )

        h_fc1 = tf.contrib.layers.fully_connected(
            inputs=h_joined,
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

        self.saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def get_action(
        self,
        state_rep,
        prev_a,
        e=0.1,
    ):
        action = self.sess.run(
            self.action,
            feed_dict={
                self.x: [state_rep],
                self.prev_a: [prev_a],
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
        prev_actions,
    ):
        q = self.sess.run(
            self.Q,
            feed_dict={
                self.x: states,
                self.prev_a: prev_actions,
            },
        )
        next_q = self.sess.run(
            self.Q,
            feed_dict={
                self.x: next_states,
                self.prev_a: actions,
            },
        )
       
        for i, r in enumerate(rewards):
            max_next_q = np.max(next_q[i,:])
            q[i, actions[i]] = max_next_q * DISCOUNT + r
        
        self.sess.run(
            [self.train_step],
            feed_dict={
                self.x: states,
                self.prev_a: prev_actions,
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
        
        #board_history = history.History(
        #    size=HISTORY_SIZE,
        #)
        #board_history.add(initial_state)
        state = initial_state
        prev_a = 0

        for j in range(MAX_EPISODE_STEPS):
            current_rep = state#board_history.get_rep()
            current_action = self.get_action(
                state_rep=current_rep,
                prev_a=prev_a,
                e=.01+.49/(1+episode_number/100),
            )

            next_state, terminal, reward = g.step(current_action)
            total_reward += reward

            #board_history.add(next_state)
            next_rep = next_state#board_history.get_rep()
            state = next_state

            self.exp_buffer.add(
                s=current_rep,
                a=current_action,
                r=reward,
                s2=next_rep,
                prev_a=prev_a,
            )

            prev_a = current_action

            if self.exp_buffer.size() > EXP_BUFFER_BATCH_SIZE:
                s_batch, a_batch, r_batch, s2_batch, prev_a_batch = self.exp_buffer.sample_batch(EXP_BUFFER_BATCH_SIZE)
                self.train_batch(
                    states=s_batch,
                    actions=a_batch,
                    rewards=r_batch,
                    next_states=s2_batch,
                    prev_actions=prev_a_batch,
                )

            if terminal:
                break

        return total_reward

    def train(
        self,
    ):
        g = game.Game(self.w, self.h)
        total_episodes_reward = 0
        
        for i in range(MAX_EPISODES):
            total_episodes_reward += self.train_episode(
                g=g,
                episode_number=i,
            )
          
            if i%100 == 0 and i > 0:
                print(
                    'Total reward for episode {episode_number}: {reward}'.format(
                        episode_number=i,
                        reward=total_episodes_reward/100 ,
                    )
                )
                total_episodes_reward=0
                if i%1000 == 0:
                    self.saver.save(self.sess, './snake.ckpt')

    def play(
        self,
    ):
        while True:
            g = game.Game(self.w, self.h)

            initial_state = g.reset()
            
            #board_history = history.History(
            #    size=HISTORY_SIZE,
            #)
            #board_history.add(initial_state)
            state = initial_state
            prev_a = 2
            
            input('Waiting for you...')
            while True:
                try:
                    g.draw_with_sleep()
                    
                    current_rep = state#board_history.get_rep()
                    action = self.get_action(
                        state_rep=current_rep,
                        prev_a=prev_a,
                        e=0,
                    )

                    next_state, terminal, reward = g.step(action)
                    state = next_state
                    prev_a = action
                    #board_history.append(next_state)

                    if terminal:
                        if reward:
                            g.draw_with_clear()
                        break

                except KeyboardInterrupt:
                    break
