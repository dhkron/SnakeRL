import os
import time
import random
import numpy as np

class P:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Game:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def start(self):
        self.score = 0
        self.snake = [
            P(0, int(self.h/2)),
            P(1, int(self.h/2)),
            P(2, int(self.h/2)),
        ]
        self.candy = self.get_new_candy()

    def get_snake_orientation(self):
        return P(
            self.snake[-1].x - self.snake[-2].x,
            self.snake[-1].y - self.snake[-2].y,
        )

    def get_real_direction(self, direction):
        orient = self.get_snake_orientation()
        if direction == 0:
           return P(
                orient.y,
                -orient.x,
            )
        elif direction == 1:
           return P(
                -orient.y,
                orient.x,
            )
        elif direction == 2:
            return orient

    def get_new_head(self, direction):
        head = self.snake[-1]
        step = self.get_real_direction(direction)
        new_head = P(
            x=head.x+step.x,
            y=head.y+step.y,
        )
        return new_head

    def get_new_candy(self):
        if self.w*self.h == len(self.snake):
            return None

        while True:
            new_candy = P(
                random.choice(range(self.w)),
                random.choice(range(self.h)),
            )
            for scale in self.snake:
                if new_candy.x == scale.x and new_candy.y == scale.y:
                    break
            else:
                return new_candy

    def step(self, direction):
        new_head = self.get_new_head(direction)

        if new_head.x < 0 or new_head.x >= self.w:
            return True, 0
        if new_head.y < 0 or new_head.y >= self.h:
            return True, 0
        for scale in self.snake[1:-2]:
            if new_head.x == scale.x and new_head.y == scale.y:
                return True, 0

        self.snake.append(new_head)
        if new_head.x == self.candy.x and new_head.y == self.candy.y:
            self.candy = self.get_new_candy()
            self.score += 1
            if self.candy is None:
                return True, 1
            else:
                return False, 1
        else:
            self.snake.pop(0)
            return False, 0

        raise Exception('Why am I here?')

    def get_board(self):
        board = np.zeros([self.h, self.w])
        if self.candy is not None:
            board[
                self.candy.y,
                self.candy.x,
            ] = 255
        for scale in self.snake:
            board[
                scale.y,
                scale.x,
            ] = 1
        board[
            self.snake[-1].y,
            self.snake[-1].x,
        ] = 2

        return board

    def draw(self):
        board = self.get_board()
        symbols = {
            0: ' ',
            1: '+',
            2: 'o',
            255: '*',
        }

        print('╔', end='')
        print('═' * (self.w * 2 + 1), end='')
        print('╗')
        for row in board:
            print('║', end=' ')
            for col in row:
                print(symbols[int(col)], end=' '),
            print('║')
        print('╚', end='')
        print('═' * (self.w * 2 + 1), end='')
        print('╝')

        print('score = {score}'.format(
            score=self.score,
        ))

    def draw_with_clear(self):
        os.system('clear')
        self.draw()
    
    def draw_with_sleep(self):
        self.draw_with_clear()
        time.sleep(.2)
