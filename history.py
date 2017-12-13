import collections
import numpy as np


class History:
    def __init__(
        self,
        size,
    ):
        self.states = collections.deque(
            maxlen=size,
        )
        self.rep_size = size

    def add(
        self,
        state,
    ):
        self.states.append(state)

    def clear(
        self,
    ):
        self.states.clear()

    def get_rep(
        self,
    ):
        while len(self.states) < self.rep_size:
            self.states.insert(
                0,
                np.zeros(self.states[0].shape),
            )

        rep = np.zeros(
            shape=self.states[0].shape,
        )
        weight = 1
        for mat in self.states: 
            rep[:,:,] += mat * weight
            weight *= 2

        return rep
