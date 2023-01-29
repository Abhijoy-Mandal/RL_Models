# This is a sample Python script.
import tensorflow as tf
from game import Sudoku
from Rubics_cube import RubicsCube
from time import sleep
import gymnasium as gym
import numpy as np
from carRacing import get_action, eval_getaction
import imageio
from PIL import Image



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    epsilon = 0
    max_attempts=1000
    report_move = 20
    new_model = tf.keras.models.load_model('./carRacing_v10_samecol.h5', compile=False)
    # env = RecordVideo(gym.make("CartPole-v1", render_mode='rgb_array'), './cartPoleVids')
    env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False, lap_complete_percent=0.95, render_mode="human")
    state, _ = env.reset()
    done = False
    total_score = 0
    actions = []
    for i in range(max_attempts):
        state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network
        q_values = new_model(state_qn)
        q_values = tf.squeeze(q_values)
        action = eval_getaction(q_values, epsilon)
        # Take action A and receive reward R and the next state S'
        env_action = action
        # if action == 1:
        #     env_action = 2
        next_state, reward, truncated, terminated, _= env.step(env_action)
        done = truncated or terminated
        total_score+=reward
        # if next_state[0] < -0.5:
        #     reward += 0
        # elif next_state[0] < -0.25:
        #     reward += 0.5
        # elif next_state[0] < 0:
        #     reward += 0.75
        # elif next_state[0] < 0.25:
        #     reward += 5
        # elif next_state[0] < 0.5:
        #     reward += 10
        # else:
        #     reward += 400
        # plt.show()
        state = next_state.copy()
        total_score+=reward
        # if i%report_move==0:
        #     print(f"Score at move {i}: {total_score}")
        if done:
            print(f"solved!! in {i} moves")
            break
    print(f"Total score: {total_score}")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
