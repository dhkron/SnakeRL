import collections
import random
import numpy as np


class ExpBuffer:
    def __init__(self, size):
        self.buffer_size = size
        self.count = 0
        self.buffer = collections.deque()

    def add(self, s, a, r, s2, prev_a):
        experience = (s, a, r, s2, prev_a)
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
        prev_a_batch= np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, s2_batch, prev_a_batch

    def clear(self):
        self.deque.clear()
        self.count = 0
