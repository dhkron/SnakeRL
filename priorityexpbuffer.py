import collections
import functools
import random
import numpy as np


class ExpBuffer:
    def __init__(self, size):
        factory = functools.partial(
            collections.deque,
            maxlen=size,
        )
        self.buffers = collections.defaultdict(
            factory,
        )

    def add(self, priority, s, a, r, s2):
        experience = (s, a, r, s2)
        self.buffers[priority].append(experience)

    def size(self):
        return sum(
            map(
                lambda key: len(self.buffers[key]),
                self.buffers,
            )
        )

    def sample_batch(self, batch_size):
        batch = []

        priorities = sorted(
            self.buffers.keys(),
            reverse=True,
        )
        per_priority = int(batch_size/len(priorities))
        for priority in priorities:
            batch += random.sample(
                self.buffers[priority],
                min(per_priority, len(self.buffers[priority]))
            )
        while len(batch) < batch_size:
            for priority in priorities:
                batch += random.sample(
                    self.buffers[priority],
                    1, 
                )

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        s2_batch = np.array([_[3] for _ in batch])

        return s_batch, a_batch, r_batch, s2_batch

    def clear(self):
        self.buffers.clear()
