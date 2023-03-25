import time
from collections import deque, namedtuple
import numpy as np
import tensorflow as tf
import random
import gymnasium as gym
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, InputLayer, Conv2D, MaxPool2D, Flatten, BatchNormalization, Rescaling
from tensorflow.keras.losses import MSE, Loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import imageio
import tensorflow_probability as tfp
tfd = tfp.distributions
from PIL import Image
from tensorflow.python.framework.ops import disable_eager_execution
import warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')
MEMORY_SIZE = 50000     # size of memory buffer
GAMMA = 0.995             # discount factor
ALPHA = 3e-4 # learning rate
POLICY_ALPHA = 3e-6
NUM_STEPS_FOR_UPDATE = 50  # perform a learning update every C time steps
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
state_size = 376
num_actions = 17
np.random.seed(1747)
tf.random.set_seed(
    174
)
random.seed(74)
max_arr = [3.14, 5.0, 5.0, 5.0, 3.14, 5.0, 3.14, 5.0, 5.0, 3.14, 5.0, 3.14, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
max_arr = np.asarray(max_arr)
max_arr = np.sqrt(max_arr)
max_arr = np.expand_dims(max_arr, axis=0)

class Mean(Loss):
  def call(self, y_true, y_pred):
    return tf.math.reduce_mean(y_true-y_pred)


q_network = Sequential([
    ### START CODE HERE ###
    Input(state_size+num_actions),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
    ### END CODE HERE ###
    ])

q_network_2 = Sequential([
    ### START CODE HERE ###
    Input(state_size+num_actions),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
    ### END CODE HERE ###
    ])

policy = Sequential([
    ### START CODE HERE ###
    Input(state_size),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_actions, activation='tanh'),
    ### END CODE HERE ###
    ])

target_policy = Sequential([
    ### START CODE HERE ###
    Input(state_size),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_actions, activation='tanh'),
    ### END CODE HERE ###
    ])

# Create the target Q^-Network
target_q_network = Sequential([
    ### START CODE HERE ###
    Input(state_size + num_actions),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
    ### END CODE HERE ###
    ])

# Create the target Q^-Network
target_q_network_2 = Sequential([
    ### START CODE HERE ###
    Input(state_size + num_actions),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
    ### END CODE HERE ###
    ])
optimizer = Adam(learning_rate=ALPHA)
optimizer_2 = Adam(learning_rate=ALPHA)
policy_optimiser = Adam(learning_rate=POLICY_ALPHA)


def gaussian_likelihood(action, pred):
    # https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/sac/policies.py
    log_std = -0.5 * tf.ones(num_actions, dtype=np.float32)
    pre_sum = -0.5 * (((action-pred)/(tf.math.exp(log_std)+1e-8))**2 + 2*log_std + tf.math.log(2*np.pi))
    return tf.cast(tf.reduce_sum(pre_sum, axis=1), 'float32')


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
    done_vals = tf.squeeze(done_vals)
    means = policy_network(next_states)
    a_prime, _ = get_action(means, 0.25)
    s_q_prime = tf.concat([next_states, a_prime], axis=1)
    target_qvals = target_q_network(s_q_prime)
    target_qvals_2 = alt_q_net(s_q_prime)
    target_qvals = tf.squeeze(tf.math.minimum(target_qvals, target_qvals_2))
    ### START CODE HERE ###
    y_targets = tf.math.add(rewards, tf.math.multiply((1 - done_vals), gamma * target_qvals))
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


def compute_policy_loss(experiences, q_network_2, q_network, policy_network):
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

    means = policy_network(states)
    mu_s = tf.squeeze(means)
    s_q = tf.concat([states, mu_s], axis=1)
    q_values = q_network(s_q)
    q_values = tf.squeeze(q_values)
    q_values2 = q_network_2(s_q)
    q_values2 = tf.squeeze(q_values2)
    loss = -1.0*tf.nn.compute_average_loss(tf.squeeze(tf.math.minimum(q_values, q_values2)))
    ### END CODE HERE ###
    return loss


def agent_learn(experiences, gamma):
    """
    Updates the weights of the Q networks.

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.

    """

    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network, target_policy, target_q_network_2)
    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, q_network.trainable_variables)
    # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients_1]
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
    # Update the weights of the q_network.


    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network_2, target_q_network_2, target_policy, target_q_network)
    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, q_network_2.trainable_variables)
    # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    # Update the weights of the q_network.
    optimizer_2.apply_gradients(zip(gradients, q_network_2.trainable_variables))


def policy_learn(experiences, gamma):
    with tf.GradientTape() as tape:
        loss = compute_policy_loss(experiences, q_network_2, q_network, policy)
    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, policy.trainable_variables)
    # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    # Update the weights of the q_network.
    policy_optimiser.apply_gradients(zip(gradients, policy.trainable_variables))


def update_target_network(q_network, target_q_network):
    tau = 0.005
    q_weights = np.array(q_network.get_weights())
    tq_weights = np.array(target_q_network.get_weights())
    for i in range(len(q_network.layers)):
        q_layer = q_network.layers[i]
        tq_layer = target_q_network.layers[i]
        tq_weights = np.array(tq_layer.get_weights()[0])
        tq_bias = np.array(tq_layer.get_weights()[1])
        weights = np.array(q_layer.get_weights()[0])
        bias = np.array(q_layer.get_weights()[1])
        target_q_network.layers[i].set_weights(
            [tau * weights + (1 - tau) * tq_weights, tau * bias + (1 - tau) * tq_bias])
    # target_q_network.set_weights(tau*q_weights + (1-tau)*tq_weights)


def get_action(means, abs_salt):
    salt = tf.random.normal([means.shape[0], num_actions], mean=0, stddev=1.0)
    salt = tf.minimum(salt, abs_salt)
    salt = tf.maximum(salt, -abs_salt)
    actions = means+salt
    actions = tf.squeeze(actions)
    actions = tf.minimum(actions, 1)
    actions = tf.maximum(actions, -1)
    log_pt = gaussian_likelihood(actions, means)
    return actions, log_pt

def check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer):
    if t%NUM_STEPS_FOR_UPDATE==0 and len(memory_buffer)>2000:
        return True
    return False


def eval_getaction(means):
    actions = tf.squeeze(means)
    actions = tf.minimum(actions, 1)
    actions = tf.maximum(actions, -1)
    return actions


def get_experiences(memory_buffer):
    sample = random.sample(memory_buffer, 1024)
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


def evaluate(eps, score, max_score):
    max_attempts = 1000
    env = gym.make("Humanoid-v4", render_mode="rgb_array")
    writer = imageio.get_writer(f'./humanoid-vids/episode_{eps}_{score}_{max_score}.mp4', fps=30)
    state, _ = env.reset()
    tot_reward = 0
    for i in range(max_attempts):
        # state = np.asarray(state).astype('float32')
        state_qn = np.expand_dims(state, axis=0)
        # print(state_qn)# state needs to be the right shape for the q_network
        dist = policy(state_qn)
        action, _ = get_action(dist, 0)
        action = tf.squeeze(action)
        env_action = 0.4*action
        # Take action A and receive reward R and the next state S'
        next_state, reward, truncated, terminated, _ = env.step(env_action)
        tot_reward += reward
        done = truncated or terminated
        img = np.asarray(Image.fromarray(np.asarray(env.render())).resize((512, 512)))
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
    num_episodes = 50000
    max_num_timesteps = 1000
    eval_freq = 100
    ep_offset=300

    max_score = float("-inf")
    total_point_history = []
    env = gym.make("Humanoid-v4")
    num_p_av = 100  # number of total points to use for averaging
    epsilon = 1.0  # initial ε value for ε-greedy policy
    # Create a memory buffer D with capacity N
    memory_buffer = deque(maxlen=MEMORY_SIZE)
    # Set the target network weights equal to the Q-Network weights
    # policy = tf.keras.models.load_model('humanoid_policy_td3_max3.h5')
    # q_network = tf.keras.models.load_model('humanoid_q1_td3_max3.h5')
    # q_network_2 = tf.keras.models.load_model('humanoid_q2_td3_max3.h5')
    target_q_network.set_weights(q_network.get_weights())
    target_q_network_2.set_weights(q_network_2.get_weights())
    target_policy.set_weights(policy.get_weights())
    policy_update_count = 1
    time_step=0
    for i in range(num_episodes):
        # Reset the environment to the initial state and get the initial state
        episode_time_start = time.time()
        state, _ = env.reset()
        # [[], [], [], []]
        # state = state.reshape((54,))
        total_points = 0
        highest = float("-inf")
        for t in range(max_num_timesteps):
            time_step+=1
            # state = np.asarray(state).astype('float32')
            state_qn = np.expand_dims(state, axis=0)

            dist = policy(state_qn)
            action, _ = get_action(dist, 0)
            action = tf.squeeze(action)
            env_action = 0.4*action
            # Take action A and receive reward R and the next state S'
            next_state, reward, truncated, terminated, _ = env.step(env_action)
            done = truncated or terminated
            # Store experience tuple (S,A,R,S') in the memory buffer.
            # We store the done variable as well for convenience.
            memory_buffer.append(experience(state, action, reward, next_state, done))
            # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
            update = check_update_conditions(time_step, NUM_STEPS_FOR_UPDATE, memory_buffer)
            if update:
                # Sample random mini-batch of experience tuples (S,A,R,S') from D
                for _ in range(10):
                    experiences = get_experiences(memory_buffer)
                    # Set the y targets, perform a gradient descent step,
                    # and update the network weights.
                    agent_learn(experiences, GAMMA)
                    if policy_update_count%2==0:
                        policy_learn(experiences, GAMMA)
                        update_target_network(policy, target_policy)
                    update_target_network(q_network, target_q_network)
                    update_target_network(q_network_2, target_q_network_2)
                    policy_update_count+=1


            state = next_state.copy()
            total_points += reward

            if done:
                break
        if total_points>max_score:
            max_score=total_points
            policy.save('humanoid_policy_td3_max4.h5')
            q_network.save('humanoid_q1_td3_max4.h5')
            q_network_2.save('humanoid_q2_td3_max4.h5')
        total_point_history.append(total_points)
        av_latest_points = np.mean(total_point_history[-num_p_av:])
        tot_ep_time = time.time()-episode_time_start
        print(f"\rEpisode {i + 1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f} episode_time:{tot_ep_time:.2f}, latest reward:{total_points}, max score:{max_score}",
              end=" ")

        if (i + 1) % 10 == 0:
            print(f"Episode {i + 1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")
            policy.save('humanoid_policy_td3_3.h5')
            q_network.save('humanoid_q1_td3_3.h5')
            q_network_2.save('humanoid_q2_td3_3.h5')
        # We will consider that the environment is solved if we get an
        # average of 200 points in the last 100 episodes.
        if (i+1)%eval_freq == 0:
            print("Evaluating...")
            evaluate(i+1+ep_offset, int(av_latest_points), int(max_score))

    tot_time = time.time() - start

    print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time / 60):.2f} min)")