import pandas as pd
import numpy as np
import random
class RubicsCube:

    def __init__(self, diff):
        self.cube = np.zeros((6, 3, 3))
        for i in range(6):
            self.cube[i, :, :] = -0.5 + i/6
        self.diff = diff
        self.solved_cube = self.cube.copy()
        for i in range(self.diff):
            action = random.randint(0, 5)
            self.step(action, system=True)
        self.curr_prob = self.cube.copy()

    def right(self):
        self.cube[4] = np.rot90(self.cube[4])
        self.cube[[0, 1, 5, 2], :, 2] = self.cube[[1, 5, 2, 0], :, 2]

    def left(self):
        self.cube[3] = np.rot90(self.cube[3])
        self.cube[[0, 1, 5, 2], :, 0] = self.cube[[2, 0, 1, 5], :, 0]

    def up(self):
        self.cube[1] = np.rot90(self.cube[1])
        self.cube[[0, 3, 5, 4], 0, :] = self.cube[[3, 5, 4, 0], 0, :]

    def down(self):
        self.cube[2] = np.rot90(self.cube[2])
        self.cube[[0, 3, 5, 4], 2, :] = self.cube[[4, 0, 3, 5], 2, :]

    def front(self):
        self.cube[0] = np.rot90(self.cube[0])
        self.cube[[1, 4, 2, 3], 2, :] = self.cube[[4, 2, 3, 1], 2, :]

    def back(self):
        self.cube[5] = np.rot90(self.cube[5])
        self.cube[[1, 4, 2, 3], 0, :] = self.cube[[3, 1, 4, 2], 0, :]

    def reset(self):
        for i in range(6):
            self.cube[i, :, :] = -0.5 + i/6
        # for i in range(self.diff):
        #     action = random.randint(0, 5)
        #     self.step(action, system=True)
        self.cube = self.curr_prob.copy()
        return self.cube

    def step(self, action, system=False):
        if action == 0:
            self.right()
        elif action == 1:
            self.left()
        elif action == 2:
            self.up()
        elif action == 3:
            self.down()
        elif action == 4:
            self.front()
        elif action == 5:
            self.back()
        elif action == 6:
            self.right()
            self.right()
            self.right()
        elif action == 7:
            self.left()
            self.left()
            self.left()
        elif action == 8:
            self.up()
            self.up()
            self.up()
        elif action == 9:
            self.down()
            self.down()
            self.down()
        elif action == 10:
            self.front()
            self.front()
            self.front()
        elif action == 11:
            self.back()
            self.back()
            self.back()
        solved = True
        num_solved = 0
        for i in range(6):
            side_complete = (not np.any(self.cube[i]-self.cube[i, 0, 0]))
            solved = solved and side_complete
            if side_complete:
                num_solved+=1
            # if side_complete and self.solved_sides[i] == 0:
            #     if not system:
            #         self.solved_sides[i] = 1
            #     num_solved+=1
        solved_true = np.equal(self.cube, self.solved_cube).astype("int32")
        face_solved = np.sum(np.sum(solved_true, axis=-1), axis=-1)[0]/9.0
        solved_cells=np.sum(solved_true)
        if solved:
            return self.cube, 400, solved, 0
        return self.cube, solved_cells/540 + face_solved + num_solved -1, solved, 0

    def __str__(self):
        return f"{self.cube}"

if __name__ == "__main__":
    env = RubicsCube(1)
    puzzle = env.reset()
    end = 0
    while not end:
        action = int(input("Enter action"))
        if action == -1:
            print(env)
        elif action == -2:
            env.reset()
            print(env)
        elif action == -3:
            end = 1
        else:
            next_state, reward,done, _ = env.step(action)
            print(reward, done)
