import time
from collections import deque, namedtuple
import numpy as np
import tensorflow as tf
import random
import gymnasium as gym
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, InputLayer, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import imageio
from PIL import Image
from tensorflow.python.framework.ops import disable_eager_execution
from game import Sudoku
from Rubics_cube import RubicsCube
import warnings
warnings.filterwarnings("ignore")
MEMORY_SIZE = 50000     # size of memory buffer
GAMMA = 0.995             # discount factor
ALPHA = 3e-3              # learning rate
NUM_STEPS_FOR_UPDATE = 64  # perform a learning update every C time steps
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
state_size = (96, 96, 3)
num_actions = 5
np.random.seed(1747)
tf.random.set_seed(
    174
)
conv_inputshape = (96, 96, 3)
random.seed(74)
q_network = Sequential([
    ### START CODE HERE ###
    InputLayer(conv_inputshape),
    Conv2D(8, 3, strides=1, activation='relu', padding="same"),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(16, 3, strides=1, activation='relu', padding="same"),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(32, 3, strides=1, activation='relu', padding="same"),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(64, 3, strides=1, activation='relu', padding="same"),
    MaxPool2D(pool_size=(4, 4), strides=4),
    Conv2D(64, 3, strides=1, activation='relu', padding="valid"),
    Flatten(),
    Dense(num_actions, activation='linear')
    ### END CODE HERE ###
    ])

target_q_network = Sequential([
    ### START CODE HERE ###
    InputLayer(conv_inputshape),
    Conv2D(8, 3, strides=1, activation='relu', padding="same"),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(16, 3, strides=1, activation='relu', padding="same"),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(32, 3, strides=1, activation='relu', padding="same"),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(64, 3, strides=1, activation='relu', padding="same"),
    MaxPool2D(pool_size=(4, 4), strides=4),
    Conv2D(64, 3, strides=1, activation='relu', padding="valid"),
    Flatten(),
    Dense(num_actions, activation='linear')
    ### END CODE HERE ###
    ])

# q_network = Sequential([
#     ### START CODE HERE ###
#     Input(state_size),
#     Dense(2, activation='relu', kernel_regularizer=regularizers.L1(0.001)),
# #    Dense(16, activation='relu', kernel_regularizer=regularizers.L1(0.001)),
#     Dense(num_actions, activation='linear', kernel_regularizer=regularizers.L1(0.001))
#     ### END CODE HERE ###
#     ])
#
# # Create the target Q^-Network
# target_q_network = Sequential([
#     ### START CODE HERE ###
#     Input(state_size),
#     Dense(2, activation='relu'),
# #    Dense(16, activation='relu'),
#     Dense(num_actions, activation='linear')
#     ### END CODE HERE ###
#     ])

# q_network_2 = Sequential([
#     ### START CODE HERE ###
#     Input(state_size),
#     Dense(32, activation='relu'),
#     Dense(32, activation='relu'),
#     Dense(num_actions, activation='linear')
#     ### END CODE HERE ###
#     ])
#
# # Create the target Q^-Network
# target_q_network_2 = Sequential([
#     ### START CODE HERE ###
#     Input(state_size),
#     Dense(32, activation='relu'),
#     Dense(32, activation='relu'),
#     Dense(num_actions, activation='linear')
#     ### END CODE HERE ###
#     ])

### START CODE HERE ###
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    ALPHA,
    decay_steps=5000,
    decay_rate=0.95,
    staircase=True)
optimizer = Adam(learning_rate=ALPHA)
optimizer_2 = Adam(learning_rate=ALPHA)
def compute_loss(experiences, gamma, q_network, target_q_network):
    """
    Calculates the loss.

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      q_network: (tf.keras.Sequential) Keras model for predicting the q_values
      target_q_network: (tf.keras.Sequential) Karas model for predicting the targets

    Returns:
      loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
            the y targets and the Q(s,a) values.
    """

    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences

    # Compute max Q^(s,a)
    target_qvals = target_q_network(next_states)
    max_qsa = tf.reduce_max(tf.squeeze(target_qvals), axis=-1)
    # target_qvals_2 = target_q_network_2(next_states)
    # max_qsa_2 = tf.reduce_max(tf.squeeze(target_qvals_2), axis=-1)

    actions = tf.expand_dims(actions, axis=0)
    # rewards = tf.expand_dims(rewards, axis=-1)
    # print(actions.shape)
    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    ### START CODE HERE ###
    y_targets = tf.math.add(rewards, tf.math.multiply((1 - done_vals), gamma * max_qsa))
    ### END CODE HERE ###
    # Get the q_values
    q_values = q_network(states)
    q_values = tf.squeeze(q_values)
    # q_values_2 = q_network_2(states)
    # q_values_2 = tf.squeeze(q_values_2)
    # q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
    #
    #                                          tf.cast(actions, tf.int32)], axis=0))
    q_values = tf.gather_nd(q_values, tf.transpose(tf.cast(actions, tf.int32)), batch_dims=1)
    # q_values_2 = tf.gather_nd(q_values_2, tf.transpose(tf.cast(actions, tf.int32)), batch_dims=1)
    # Compute the loss
    ### START CODE HERE ###
    loss = MSE(q_values, y_targets)
    # loss_2 = MSE(q_values_2, y_targets)
    ### END CODE HERE ###

    return loss


def update_target_network(q_network, target_q_network):
    tau = 0.005
    q_weights = np.array(q_network.get_weights())
    tq_weights = np.array(target_q_network.get_weights())
    # for i in range(0, len(q_network.layers), 2):
    #     q_layer = q_network.layers[i]
    #     tq_layer = target_q_network.layers[i]
    #     tq_weights = np.array(tq_layer.get_weights()[0])
    #     tq_bias = np.array(tq_layer.get_weights()[1])
    #     weights = np.array(q_layer.get_weights()[0])
    #     bias = np.array(q_layer.get_weights()[1])
    #     target_q_network.layers[i].set_weights([tau*weights+(1-tau)*tq_weights, tau*bias+(1-tau)*tq_bias])
    #     print("here")
    target_q_network.set_weights(tau*q_weights + (1-tau)*tq_weights)


@tf.function
def agent_learn(experiences, gamma):
    """
    Updates the weights of the Q networks.

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.

    """

    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)
    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, q_network.trainable_variables)

    # Update the weights of the q_network.
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
    #
    # gradients = tape.gradient(loss_2, q_network_2.trainable_variables)
    # optimizer_2.apply_gradients(zip(gradients, q_network_2.trainable_variables))
    # update the weights of target q_network


def get_action(q_values, i):
    # s = random.random()
    q_values = q_values - np.max(q_values)
    p_as = np.exp(q_values) / np.sum(np.exp(q_values))
    # print(p_as)
    # if i<=100:
    #     p_as = [1.0/num_actions]*num_actions
    action_key = np.random.choice(a=num_actions, p=p_as)
    return action_key
    # if s<=epsilon:
    #     return random.randint(0, num_actions-1)
    # else:
    #     return tf.math.argmax(q_values, 0).numpy()

def eval_getaction(q_values, i):
    return np.argmax(q_values, axis=0)

def check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer):
    if t%NUM_STEPS_FOR_UPDATE==0 and len(memory_buffer)>128:
        return True
    return False


def get_experiences(memory_buffer):
    sample = random.sample(memory_buffer, 128)
    states = []
    actions = []
    rewards = []
    next_states = []
    done_vals = []
    for exp in sample:
        state, action, reward, next_state, done_val = exp
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        done_vals.append(done_val)
    return np.array(states).astype('float32'), np.array(actions).astype('float32'), np.array(rewards).astype('float32'), np.array(next_states).astype('float32'), np.array(done_vals).astype('float32')


def get_new_eps(epsilon):
    return max(0.9985*epsilon, 0.01)


def evaluate(eps, score):
    epsilon = 0
    max_attempts = 1000
    env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False, lap_complete_percent=0.95, render_mode="rgb_array")
    writer = imageio.get_writer(f'./raceCarVids9/episode_{eps}_{score}.mp4', fps=60)
    state, _ = env.reset()
    for i in range(max_attempts):
        # state = np.asarray(state).astype('float32')
        # state = state / 255.0
        state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network
        q_values = q_network(state_qn)
        q_values = tf.squeeze(q_values)
        action = get_action(q_values, 1000)
        # Take action A and receive reward R and the next state S'
        env_action = action
        next_state, reward, done, _, _ = env.step(env_action)
        img = np.asarray(Image.fromarray(np.asarray(env.render())).resize((1920, 1280)))
        writer.append_data(img)
        state = next_state.copy()
        if done:
            print(f"solved!! in {i} moves")
            break
    writer.close()
    env.close()


if __name__=='__main__':
    start = time.time()
    num_episodes = 10000
    max_num_timesteps = 1000
    eval_freq = 50
    total_point_history = []
    q_network = tf.keras.models.load_model('./carRacing_v12_samecol.h5')
    env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False, lap_complete_percent=0.95)
    num_p_av = 50  # number of total points to use for averaging
    epsilon = 1.0  # initial ε value for ε-greedy policy
    action_list = [0]*num_actions
    # Create a memory buffer D with capacity N
    memory_buffer = deque(maxlen=MEMORY_SIZE)
    # Set the target network weights equal to the Q-Network weights
    target_q_network.set_weights(q_network.get_weights())

    for i in range(num_episodes):

        # Reset the environment to the initial state and get the initial state
        episode_time_start = time.time()
        state, _ = env.reset()
        # state = state.reshape((54,))
        total_points = 0
        highest = float("-inf")

        for t in range(max_num_timesteps):
            # state = np.asarray(state).astype('float32')
            # state = state/255.0

            state_qn = np.expand_dims(state, axis=0)
            # print(state_qn)# state needs to be the right shape for the q_network
            q_values = q_network(state_qn)
            q_values = tf.squeeze(q_values)
            action = get_action(q_values, i)
            action_list[action] += 1
            env_action = action
            # Take action A and receive reward R and the next state S'
            next_state, reward, truncated, terminated, _ = env.step(env_action)
            done = truncated or terminated
            reward = reward*10.0
            # Store experience tuple (S,A,R,S') in the memory buffer.
            # We store the done variable as well for convenience.
            memory_buffer.append(experience(state, action, reward, next_state, done))
            # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
            update = check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)
            if update:
                # Sample random mini-batch of experience tuples (S,A,R,S') from D
                experiences = get_experiences(memory_buffer)
                # Set the y targets, perform a gradient descent step,
                # and update the network weights.
                agent_learn(experiences, GAMMA)
                update_target_network(q_network, target_q_network)
                # update_target_network(q_network_2, target_q_network_2)

            state = next_state.copy()
            total_points += reward

            if done:
                break
        total_point_history.append(total_points)
        av_latest_points = np.mean(total_point_history[-num_p_av:])
        # Update the ε value
        epsilon = get_new_eps(epsilon)
        tot_ep_time = time.time()-episode_time_start
        print(f"\rEpisode {i + 1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f} episode_time:{tot_ep_time:.2f}, actions: {action_list}, latest reward:{total_points}",
              end="")

        if (i + 1) % num_p_av == 0:
            print(f"Episode {i + 1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")
            print(f"action freq: {action_list}")
            action_list = [0] * num_actions
            q_network.save('carRacing_v13_samecol.h5')
        # We will consider that the environment is solved if we get an
        # average of 200 points in the last 100 episodes.
        if (i+1)%eval_freq == 0:
            print("Evaluating...")
            evaluate(i+1, int(av_latest_points))
        # if av_latest_points >100:
        #     max_num_timesteps = 200
        # if av_latest_points >= 250.0 and len(total_point_history)>100:
        #     print(f"\n\nEnvironment solved in {i + 1} episodes!")
        #     q_network.save('carRacing_v6_samecol.h5')
        #     break

    tot_time = time.time() - start

    print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time / 60):.2f} min)")