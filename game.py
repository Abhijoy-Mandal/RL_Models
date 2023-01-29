import pandas as pd
import numpy as np
import random
class Sudoku:

    def __init__(self, diff):
        self.df = pd.read_csv('./sudoku_mini.csv', low_memory=False, dtype=str)
        self.curr_puzzle = {}
        self.diff = diff
        self.puzzle_rec = self.df.sample(n=1)

    def reset(self):
        record = self.puzzle_rec
        puzzle = [int(ch) for ch in record["puzzle"].values[0]]
        original_puzzle = [int(ch) for ch in record["puzzle"].values[0]]
        solution = [int(ch) for ch in record["solution"].values[0]]
        puzzle = np.array(puzzle).reshape((9, 9))
        original_puzzle = np.array(original_puzzle).reshape((9, 9))
        solution = np.array(solution).reshape((9, 9))
        puzz = []
        orig = []
        sol = []
        for i in range(9):
            puzz.append(np.equal(puzzle, i + 1, dtype='float32'))
            orig.append(np.equal(original_puzzle, i + 1, dtype='float32'))
            sol.append(np.equal(solution, i + 1, dtype='float32'))
        self.curr_puzzle["puzzle"] = np.array(puzz).astype('float32')
        self.curr_puzzle["original_puzzle"] = np.array(orig).astype('float32')
        self.curr_puzzle["solution"] = np.array(sol).astype('float32')

        return self.curr_puzzle["puzzle"]

    def reset3(self):
        record = self.df.sample(n=1)
        diff = random.randint(1, self.diff)
        empty_vals = random.sample(range(81), diff)
        solution = [int(ch) for ch in record["solution"].values[0]]
        puzzle = solution[:]
        original_puzzle = solution[:]
        for idx in empty_vals:
            puzzle[idx] = 0
            original_puzzle[idx] = 0
        puzzle = np.array(puzzle).reshape((9, 9))
        original_puzzle = np.array(original_puzzle).reshape((9, 9))
        solution = np.array(solution).reshape((9, 9))
        puzz = []
        orig = []
        sol = []
        for i in range(9):
            puzz.append(np.equal(puzzle, i + 1, dtype='float32'))
            orig.append(np.equal(original_puzzle, i + 1, dtype='float32'))
            sol.append(np.equal(solution, i + 1, dtype='float32'))
        self.curr_puzzle["puzzle"] = np.array(puzz).astype('float32')
        self.curr_puzzle["original_puzzle"] = np.array(orig).astype('float32')
        self.curr_puzzle["solution"] = np.array(sol).astype('float32')

        return self.curr_puzzle["puzzle"]

    def reset2(self):
        record = self.df.sample(n=1)
        puzzle = [int(ch) for ch in record["puzzle"].values[0]]
        original_puzzle = [int(ch) for ch in record["puzzle"].values[0]]
        solution = [int(ch) for ch in record["solution"].values[0]]
        puzzle = np.array(puzzle).reshape((9, 9))
        original_puzzle = np.array(original_puzzle).reshape((9, 9))
        solution = np.array(solution).reshape((9, 9))
        puzz = []
        orig = []
        sol = []
        for i in range(9):
            puzz.append(np.equal(puzzle, i+1, dtype='float32'))
            orig.append(np.equal(original_puzzle, i+1, dtype='float32'))
            sol.append(np.equal(solution, i+1, dtype='float32'))
        self.curr_puzzle["puzzle"] = np.array(puzz).astype('float32')
        self.curr_puzzle["original_puzzle"] = np.array(orig).astype('float32')
        self.curr_puzzle["solution"] = np.array(sol).astype('float32')

        return self.curr_puzzle["puzzle"]

    def step(self, action):
        try:
            action = int(action)
        except:
            print(action)
        val = action%9
        pos = action//9
        if np.any(self.curr_puzzle["original_puzzle"][:, pos//9, pos%9]):
            return self.curr_puzzle["puzzle"], -5, False, 0
        if self.curr_puzzle["original_puzzle"][val, pos//9, pos%9] != 0:
            return self.curr_puzzle["puzzle"], -5, False, 0
        if self.curr_puzzle["puzzle"][val, pos//9, pos%9] == 1:
            return self.curr_puzzle["puzzle"], -5, False, 0
        self.curr_puzzle["puzzle"][:, pos//9, pos%9] = 0
        self.curr_puzzle["puzzle"][val, pos//9, pos%9] = 1
        if np.array_equal(self.curr_puzzle["puzzle"], self.curr_puzzle["solution"]):
            return self.curr_puzzle["puzzle"], 100, True, 0
        if self.curr_puzzle["solution"][val, pos//9, pos%9] == 1:
            return self.curr_puzzle["puzzle"], 10, False, 0
        return self.curr_puzzle["puzzle"], -0.3, False, 0

    def __str__(self):
        curr_state = np.zeros((9, 9))
        solution = np.zeros((9, 9))
        for i in range(9):
            curr_state = curr_state + self.curr_puzzle['puzzle'][i]*(i+1)
            solution = solution + self.curr_puzzle['solution'][i]*(i+1)
        return f"current state: {curr_state}\nsolution: {solution}"


if __name__ == "__main__":
    env = Sudoku(1)
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
            next_state, reward, done, _ = env.step(action)
            print(next_state, reward, done)
