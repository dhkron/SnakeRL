import collections
import numpy as np


class History:
    def __init__(
        self,
        num_channels,
    ):
        self.states = collections.deque(
            maxlen=num_channels,
        )
        self.rep_size = num_channels

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

        shape = [
            self.states[0].shape[0],
            self.states[0].shape[1],
            self.rep_size,
        ]

        rep = np.ndarray(
            shape=shape,
        )
        for i in range(self.rep_size):
            rep[:,:,i] = self.states[i]

        return rep
