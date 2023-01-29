import time
from collections import deque, namedtuple
import numpy as np
import tensorflow as tf
import random
import math
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, InputLayer, Conv2D
from tensorflow.keras.losses import MSE, Loss
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.ops import disable_eager_execution
from game import Sudoku
from Rubics_cube import RubicsCube
import warnings
warnings.filterwarnings("ignore")
MEMORY_SIZE = 100000     # size of memory buffer
GAMMA = 0.9995             # discount factor
ALPHA = 1e-4              # learning rate
exploartion_coeff = 0.005
NUM_STEPS_FOR_UPDATE = 8  # perform a learning update every C time steps
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
state_size = 6*3*3
num_actions = 6
tf.random.set_seed(
    666
)
conv_inputshape = (9, 9, 9)


class MeanError(Loss):
  def call(self, y_true, y_pred):
    return tf.reduce_mean(y_pred - y_true, axis=0)
# q_network = Sequential([
#     ### START CODE HERE ###
#     InputLayer(conv_inputshape),
#     Conv2D(64, 3, strides=(3, 3), activation='relu'),
#     Conv2D(256,3, activation='relu'),
#     Conv2D(num_actions, 1, activation='linear')
#     ### END CODE HERE ###
#     ])
#
# target_q_network = Sequential([
#     ### START CODE HERE ###
#     InputLayer(conv_inputshape),
#     Conv2D(64, 3, strides=(3, 3), activation='relu'),
#     Conv2D(256,3, activation='relu'),
#     Conv2D(num_actions, 1, activation='linear')
#     ### END CODE HERE ###
#     ])

q_network = Sequential([
    ### START CODE HERE ###
    Input(state_size),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_actions, activation='linear')
    ### END CODE HERE ###
    ])

# Create the target Q^-Network
target_q_network = Sequential([
    ### START CODE HERE ###
    Input(state_size),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_actions, activation='linear')
    ### END CODE HERE ###
    ])

policy_network = Sequential([
    ### START CODE HERE ###
    Input(state_size),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_actions, activation='softmax')
    ### END CODE HERE ###
    ])

### START CODE HERE ###
optimizer = Adam(learning_rate=ALPHA)
policy_optimizer = Adam(learning_rate=ALPHA)

def compute_loss(experiences, gamma, q_network, target_q_network, policy_network):
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
    actions = tf.expand_dims(actions, axis=0)
    # Compute max Q^(s,a)
    target_qvals = target_q_network(next_states)
    max_qsa = tf.reduce_max(tf.squeeze(target_qvals), axis=-1)
    # policy = 0.001 * tf.math.log(tf.squeeze(policy_network(states)))
    # q_values = tf.gather_nd(policy, tf.transpose(tf.cast(actions, tf.int32)), batch_dims=1)
    # rewards = tf.expand_dims(rewards, axis=-1)
    # print(actions.shape)
    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    ### START CODE HERE ###
    y_targets = tf.math.add(rewards, tf.math.multiply((1 - done_vals), gamma * max_qsa))
    ### END CODE HERE ###
    # Get the q_values
    q_values = q_network(states)
    q_values = tf.squeeze(q_values)
    # q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
    #
    #                                          tf.cast(actions, tf.int32)], axis=0))
    q_values = tf.gather_nd(q_values, tf.transpose(tf.cast(actions, tf.int32)), batch_dims=1)
    # Compute the loss
    ### START CODE HERE ###
    loss = MSE(q_values, y_targets)
    ### END CODE HERE ###

    return loss


def policy_loss(experiences, q_network, policy_network):
    states, actions, rewards, next_states, done_vals = experiences
    q_values = q_network(states)
    q_values = tf.squeeze(q_values)
    # tf.print(q_values)
    policy = 0.001*tf.math.log(tf.squeeze(policy_network(states)))
    # tf.print(policy)
    loss = MeanError().call(q_values, policy)
    return loss

def update_target_network(q_network, target_q_network):
    tau = 0.001
    q_weights = q_network.get_weights()
    tq_weights = target_q_network.get_weights()
    for i in range(len(q_network.layers)):
        q_layer = q_network.layers[i]
        tq_layer = target_q_network.layers[i]
        tq_weights = np.array(tq_layer.get_weights()[0])
        tq_bias = np.array(tq_layer.get_weights()[1])
        weights = np.array(q_layer.get_weights()[0])
        bias = np.array(q_layer.get_weights()[1])
        target_q_network.layers[i].set_weights([tau*weights+(1-tau)*tq_weights, tau*bias+(1-tau)*tq_bias])

    # target_q_network.set_weights(tau*q_weights + (1-tau)*tq_weights)


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
        loss = compute_loss(experiences, gamma, q_network, target_q_network, policy_network)
    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, q_network.trainable_variables)

    # Update the weights of the q_network.
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    # update the weights of target q_network

@tf.function
def policy_learn(experiences):
    """
    Updates the weights of the Q networks.

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.

    """

    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = policy_loss(experiences, q_network, policy_network)
    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, policy_network.trainable_variables)

    # Update the weights of the q_network.
    policy_optimizer.apply_gradients(zip(gradients, policy_network.trainable_variables))


def get_action(q_values, epsilon):
    s = random.random()
    if s<=epsilon:
        return tf.math.argmax(q_values, 0).numpy()
    else:
        return tf.math.argmax(q_values, 0).numpy()


def check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer):
    if t%NUM_STEPS_FOR_UPDATE==0 and len(memory_buffer)>32:
        return True
    return False


def get_experiences(memory_buffer):
    sample = random.sample(memory_buffer, 32)
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



if __name__=='__main__':
    start = time.time()
    num_episodes = 20000
    max_num_timesteps = 100
    total_point_history = []
    env = RubicsCube(25)
    num_p_av = 100  # number of total points to use for averaging
    epsilon = 1.0  # initial ε value for ε-greedy policy

    # Create a memory buffer D with capacity N
    memory_buffer = deque(maxlen=MEMORY_SIZE)
    # Set the target network weights equal to the Q-Network weights
    target_q_network.set_weights(q_network.get_weights())

    for i in range(num_episodes):

        # Reset the environment to the initial state and get the initial state
        state = env.reset()
        state = state.reshape((54,))
        total_points = 0

        for t in range(max_num_timesteps):

            # From the current state S choose an action A using an ε-greedy policy
            state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network
            q_values = q_network(state_qn)
            prob = tf.squeeze(policy_network(state_qn))
            q_values = tf.squeeze(q_values)
            action = get_action(prob, epsilon)
            # Take action A and receive reward R and the next state S'
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape((54,))
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
                policy_learn(experiences)

            state = next_state.copy()
            total_points += reward

            if done:
                break

        total_point_history.append(total_points)
        av_latest_points = np.mean(total_point_history[-num_p_av:])
        # Update the ε value

        print(f"\rEpisode {i + 1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}",
              end="")

        if (i + 1) % num_p_av == 0:
            print(f"Episode {i + 1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")
            q_network.save('Rubicscube_SAC_v7.h5')
        # We will consider that the environment is solved if we get an
        # average of 200 points in the last 100 episodes.

        if av_latest_points >= 100.0 and len(total_point_history)>100:
            print(f"\n\nEnvironment solved in {i + 1} episodes!")
            q_network.save('Rubicscube_SAC_v7.h5')
            break

    tot_time = time.time() - start

    print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time / 60):.2f} min)")