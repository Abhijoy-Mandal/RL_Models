import time
from collections import deque, namedtuple
import numpy as np
import tensorflow as tf
import random
import gymnasium as gym
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, InputLayer, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.losses import MSE, Loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import imageio
import tensorflow_probability as tfp
tfd = tfp.distributions
from PIL import Image
from tensorflow.python.framework.ops import disable_eager_execution
from game import Sudoku
from Rubics_cube import RubicsCube
import warnings
warnings.filterwarnings("ignore")
MEMORY_SIZE = 50000     # size of memory buffer
GAMMA = 0.95             # discount factor
ALPHA = 3e-4 # learning rate
_alpha_lr = 3e-4
POLICY_ALPHA = 3e-4
NUM_STEPS_FOR_UPDATE = 64  # perform a learning update every C time steps
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
state_size = 24
num_actions = 4
np.random.seed(1747)
tf.random.set_seed(
    174
)
random.seed(74)
max_arr = [3.14, 5.0, 5.0, 5.0, 3.14, 5.0, 3.14, 5.0, 5.0, 3.14, 5.0, 3.14, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
max_arr = np.asarray(max_arr)
max_arr = np.sqrt(max_arr)
max_arr = np.expand_dims(max_arr, axis=0)

_log_alpha = tf.Variable(0.0)
_alpha = tfp.util.DeferredTensor(_log_alpha, tf.identity)
_alpha_optimizer = tf.optimizers.Adam(
            _alpha_lr, name='alpha_optimizer')

class Mean(Loss):
  def call(self, y_true, y_pred):
    return tf.math.reduce_mean(y_true-y_pred)


q_network = Sequential([
    ### START CODE HERE ###
    Input(state_size+num_actions),
    Dense(16, activation='relu', kernel_regularizer=regularizers.L1(0.001)),
    Dense(8, activation='relu', kernel_regularizer=regularizers.L1(0.001)),
    Dense(1, activation='linear', kernel_regularizer=regularizers.L1(0.001))
    ### END CODE HERE ###
    ])

q_network_2 = Sequential([
    ### START CODE HERE ###
    Input(state_size+num_actions),
    Dense(16, activation='relu', kernel_regularizer=regularizers.L1(0.001)),
    Dense(8, activation='relu', kernel_regularizer=regularizers.L1(0.001)),
    Dense(1, activation='linear', kernel_regularizer=regularizers.L1(0.001))
    ### END CODE HERE ###
    ])

policy = Sequential([
    ### START CODE HERE ###
    Input(state_size),
    Dense(16, activation='relu', kernel_regularizer=regularizers.L1(0.01)),
    Dense(8, activation='relu', kernel_regularizer=regularizers.L1(0.01)),
    Dense(num_actions, activation='tanh', kernel_regularizer=regularizers.L1(0.01))
    ### END CODE HERE ###
    ])

# Create the target Q^-Network
target_q_network = Sequential([
    ### START CODE HERE ###
    Input(state_size+num_actions),
    Dense(16, activation='relu', kernel_regularizer=regularizers.L1(0.001)),
    Dense(8, activation='relu', kernel_regularizer=regularizers.L1(0.001)),
    Dense(1, activation='linear', kernel_regularizer=regularizers.L1(0.001))
    ### END CODE HERE ###
    ])

# Create the target Q^-Network
target_q_network_2 = Sequential([
    ### START CODE HERE ###
    Input(state_size+num_actions),
    Dense(16, activation='relu', kernel_regularizer=regularizers.L1(0.001)),
    Dense(8, activation='relu', kernel_regularizer=regularizers.L1(0.001)),
    Dense(1, activation='linear', kernel_regularizer=regularizers.L1(0.001))
    ### END CODE HERE ###
    ])

### START CODE HERE ###
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    ALPHA,
    decay_steps=5000,
    decay_rate=0.95,
    staircase=True)
optimizer = Adam(learning_rate=ALPHA)
optimizer_2 = Adam(learning_rate=ALPHA)
policy_optimiser = Adam(learning_rate=POLICY_ALPHA)
def compute_loss(experiences, gamma, q_network, target_q_network, policy_network, alt_q_net):
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
    means = policy_network(next_states)
    dist = tfd.Normal(loc=means, scale=1.0)
    a_prime = eval_getaction(means)
    action_probs = tf.math.log(dist.prob(a_prime))
    action_probs = tf.math.reduce_mean(action_probs, axis=1)

    # action_probs = tf.math.reduce_max(entropy, axis=1)
    entropy_scale = tf.convert_to_tensor(_alpha)
    entropy = -1.0*entropy_scale*action_probs
    s_q_prime = tf.concat([next_states, a_prime], axis=1)
    target_qvals = target_q_network(s_q_prime)
    target_qvals_2 = alt_q_net(s_q_prime)
    target_qvals = tf.math.minimum(target_qvals, target_qvals_2)

    ### START CODE HERE ###
    y_targets = tf.math.add(tf.math.add(rewards, tf.math.multiply((1 - done_vals), gamma * target_qvals)), entropy)
    ### END CODE HERE ###
    # Get the q_values
    s_q = tf.concat([states, actions], axis=1)
    q_values = q_network(s_q)
    q_values = tf.squeeze(q_values)
    # Compute the loss
    ### START CODE HERE ###
    mse = tf.keras.losses.MeanSquaredError()
    loss = mse(q_values, y_targets)
    # loss_2 = MSE(q_values_2, y_targets)
    ### END CODE HERE ###
    return loss

def compute_policy_loss(experiences, gamma, q_network, q_network_2, policy_network):
    """
    Calculates the loss.

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      q_network: (tf.keras.Sequential) Keras model for predicting the q_values

    Returns:
      loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
            the y targets and the Q(s,a) values.
    """

    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences

    # Compute max Q^(s,a)
    ### END CODE HERE ###
    # Get the q_values
    s_q = tf.concat([states, actions], axis=1)
    q_values = q_network(s_q)
    q_values = tf.squeeze(q_values)
    q_values_2 = q_network_2(s_q)
    q_values_2 = tf.squeeze(q_values_2)
    q_values = tf.math.minimum(q_values_2, q_values)
    # Compute the loss
    ### START CODE HERE ###
    means = policy_network(next_states)
    dist = tfd.Normal(loc=means, scale=1.0)
    a_prime = eval_getaction(means)
    action_probs = tf.math.log(dist.prob(a_prime))
    action_probs = tf.math.reduce_mean(action_probs, axis=1)

    # action_probs = tf.math.reduce_max(entropy, axis=1)
    entropy_scale = tf.convert_to_tensor(_alpha)
    policy_losses = entropy_scale * action_probs - q_values
    loss = tf.nn.compute_average_loss(policy_losses)
    # Get the gradients of the loss with respect to the weights.
    # loss_2 = MSE(q_values_2, y_targets)
    ### END CODE HERE ###
    return loss


def compute_alpha_loss(experiences, policy_network):
    states, actions, rewards, next_states, done_vals = experiences
    means = policy_network(next_states)
    dist = tfd.Normal(loc=means, scale=1.0)
    a_prime = eval_getaction(means)
    action_probs = tf.math.log(dist.prob(a_prime))
    action_probs = tf.math.reduce_mean(action_probs, axis=1)
    # entropy = dist.entropy()
    # Compute max Q^(s,a)
    # action_probs = tf.math.reduce_prod(dist.prob(a_prime), axis=1)
    # action_probs = tf.math.log(tf.math.maximum(action_probs, 1e-5))
    # action_probs = tf.math.reduce_max(entropy, axis=1)
    # tf.print(tf.math.reduce_mean(entropy))
    alpha_losses = -1.0*(
            _alpha * tf.stop_gradient(action_probs + 2))
    alpha_loss = tf.nn.compute_average_loss(alpha_losses)
    return alpha_loss


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
        loss = compute_loss(experiences, gamma, q_network, target_q_network, policy, target_q_network_2)
    # Get the gradients of the loss with respect to the weights.
    gradients_1 = tape.gradient(loss, q_network.trainable_variables)

    # Update the weights of the q_network.


    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network_2, target_q_network_2, policy, target_q_network)
    # Get the gradients of the loss with respect to the weights.
    optimizer.apply_gradients(zip(gradients_1, q_network.trainable_variables))
    gradients = tape.gradient(loss, q_network_2.trainable_variables)

    # Update the weights of the q_network.
    optimizer_2.apply_gradients(zip(gradients, q_network_2.trainable_variables))

    with tf.GradientTape() as tape:
        loss = compute_policy_loss(experiences, gamma, q_network, q_network_2, policy)
    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, policy.trainable_variables)

    # Update the weights of the q_network.
    policy_optimiser.apply_gradients(zip(gradients, policy.trainable_variables))

    with tf.GradientTape() as tape:
        loss = compute_alpha_loss(experiences, policy)
    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, [_log_alpha])

    # Update the weights of the q_network.
    _alpha_optimizer.apply_gradients(zip(gradients, [_log_alpha]))
    #
    # gradients = tape.gradient(loss_2, q_network_2.trainable_variables)
    # optimizer_2.apply_gradients(zip(gradients, q_network_2.trainable_variables))
    # update the weights of target q_network


def get_action(means):
    actions = tf.random.normal([means.shape[0], num_actions], mean=means, stddev=_alpha)
    actions = tf.squeeze(actions)
    actions = tf.minimum(actions, 1.0)
    actions = tf.maximum(actions, -1.0)
    return actions

def eval_getaction(means):
    actions = tf.squeeze(means)
    actions = tf.minimum(actions, 1.0)
    actions = tf.maximum(actions, -1.0)
    return actions

def check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer):
    if t%NUM_STEPS_FOR_UPDATE==0 and len(memory_buffer)>64:
        return True
    return False


def get_experiences(memory_buffer):
    sample = random.sample(memory_buffer, 64)
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
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    writer = imageio.get_writer(f'./walker_Vids7/episode_{eps}_{score}.mp4', fps=60)
    state, _ = env.reset()
    tot_reward = 0
    for i in range(max_attempts):
        # state = np.asarray(state).astype('float32')
        # state = state / 255.0
        state_qn = np.expand_dims(state, axis=0)
        # print(state_qn)# state needs to be the right shape for the q_network
        dist = policy(state_qn)
        action = eval_getaction(dist)
        action = tf.squeeze(action)
        env_action = action
        # Take action A and receive reward R and the next state S'
        next_state, reward, truncated, terminated, _ = env.step(env_action)
        tot_reward += reward
        done = truncated or terminated
        img = np.asarray(Image.fromarray(np.asarray(env.render())).resize((1920, 1280)))
        writer.append_data(img)
        state = next_state.copy()
        if done:
            print(f"solved!! in {i} moves; score:{tot_reward}")
            break
    print(f"score:{tot_reward}")
    writer.close()
    env.close()


if __name__=='__main__':
    start = time.time()
    num_episodes = 10000
    max_num_timesteps = 1000
    eval_freq = 50
    total_point_history = []
    env = gym.make("BipedalWalker-v3")
    num_p_av = 100  # number of total points to use for averaging
    epsilon = 1.0  # initial ε value for ε-greedy policy
    # Create a memory buffer D with capacity N
    memory_buffer = deque(maxlen=MEMORY_SIZE)
    # Set the target network weights equal to the Q-Network weights
    target_q_network.set_weights(q_network.get_weights())
    target_q_network_2.set_weights(q_network_2.get_weights())
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
            dist = policy(state_qn)
            action = eval_getaction(dist)
            action = tf.squeeze(action)
            env_action = action
            # Take action A and receive reward R and the next state S'
            next_state, reward, truncated, terminated, _ = env.step(env_action)
            reward = reward*100
            done = truncated or terminated
            # reward = reward*10
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
                update_target_network(q_network_2, target_q_network_2)

            state = next_state.copy()
            total_points += reward

            if done:
                break
        total_point_history.append(total_points)
        av_latest_points = np.mean(total_point_history[-num_p_av:])
        # Update the ε value
        epsilon = get_new_eps(epsilon)
        tot_ep_time = time.time()-episode_time_start
        print(f"\rEpisode {i + 1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f} episode_time:{tot_ep_time:.2f}, latest reward:{total_points}",
              end=" ")

        if (i + 1) % num_p_av == 0:
            print(f"Episode {i + 1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")
            policy.save('walker_v5.h5')
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