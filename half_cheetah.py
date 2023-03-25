import time
from collections import deque, namedtuple
import numpy as np
import pandas
import tensorflow as tf
import random
import gymnasium as gym
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, InputLayer, Conv2D, MaxPool2D, Flatten, Rescaling
from tensorflow.keras.losses import MSE, Loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import imageio
import copy
import tensorflow_probability as tfp
tfd = tfp.distributions
from PIL import Image
from tensorflow.python.framework.ops import disable_eager_execution
from game import Sudoku
from Rubics_cube import RubicsCube
import warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

MEMORY_SIZE = 100000     # size of memory buffer
GAMMA = 0.95             # discount factor
ALPHA = 3e-4 # learning rate
POLICY_ALPHA = 3e-4
NUM_STEPS_FOR_UPDATE = 8  # perform a learning update every C time steps
experience = namedtuple("Experience", field_names=["states", "actions", "values", "next_values", "dones", "advantages", "targets", "logp_ts"])
state_size = 17
num_actions = 6
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


critic_network = Sequential([
    ### START CODE HERE ###
    Input(state_size),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
    ### END CODE HERE ###
    ])

actor_network = Sequential([
    ### START CODE HERE ###
    Input(state_size),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(num_actions, activation='tanh')
    ### END CODE HERE ###
    ])

### START CODE HERE ###
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    ALPHA,
    decay_steps=5000,
    decay_rate=0.95,
    staircase=True)
optimizer = Adam(learning_rate=ALPHA, clipnorm=0.25)
policy_optimiser = Adam(learning_rate=POLICY_ALPHA, clipnorm=0.25)


def gaussian_likelihood(action, pred):
    # https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/sac/policies.py
    log_std = -0.5 * tf.ones(num_actions, dtype=np.float32)
    pre_sum = -0.5 * (((action-pred)/(tf.math.exp(log_std)+1e-8))**2 + 2*log_std + tf.math.log(2*np.pi))
    return tf.cast(tf.reduce_sum(pre_sum, axis=1), 'float32')


def compute_loss(experiences, critic_network):
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
    states, actions, values, next_values, done_vals, advantages, targets, log_old_pts = experiences
    LOSS_CLIPPING = 0.2
    y_pred = critic_network(states)
    y_true = tf.cast(targets, "float32")
    clipped_value_loss = tf.cast(values + tf.clip_by_value(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING), "float32")
    v_loss1 = tf.squeeze((y_true - clipped_value_loss) ** 2)
    v_loss2 = tf.squeeze((y_true - y_pred) ** 2)
    value_loss = 0.5 * tf.nn.compute_average_loss(tf.maximum(v_loss1, v_loss2))
    return value_loss


def compute_policy_loss(experiences, actor_network):
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
    states, actions, values, next_values, done_vals, advantages, targets, log_old_pts = experiences
    y_pred = actor_network(states)
    LOSS_CLIPPING = 0.2
    logp = gaussian_likelihood(actions, y_pred)
    ratio = tf.cast(tf.math.exp(logp - log_old_pts), 'float32')
    advantages=tf.cast(advantages, 'float32')
    advantages = tf.squeeze(advantages)
    p1 = ratio * advantages
    p2 = tf.where(advantages > 0, (1.0 + LOSS_CLIPPING) * advantages,
                  (1.0 - LOSS_CLIPPING) * advantages)  # minimum advantage
    actor_loss = -tf.nn.compute_average_loss(tf.minimum(p1, p2))
    return actor_loss


@tf.function
def critic_learn(experiences):
    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, critic_network)
    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, critic_network.trainable_variables)
    # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients_1]
    optimizer.apply_gradients(zip(gradients, critic_network.trainable_variables))
    # Update the weights of the q_network.


@tf.function
def actor_learn(experiences):
    """
    Updates the weights of the Q networks.

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.

    """

    with tf.GradientTape() as tape:
        loss = compute_policy_loss(experiences, actor_network)
    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, actor_network.trainable_variables)
    # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    # Update the weights of the q_network.
    policy_optimiser.apply_gradients(zip(gradients, actor_network.trainable_variables))


def get_action(means):
    salt = tf.random.normal([means.shape[0], num_actions], mean=0, stddev=1.0)
    salt = tf.minimum(salt, 0.2)
    salt = tf.maximum(salt, -0.2)
    actions = means+salt
    actions = tf.squeeze(actions)
    actions = tf.minimum(actions, 1)
    actions = tf.maximum(actions, -1)
    log_pt = gaussian_likelihood(actions, means)
    return actions, log_pt

def eval_getaction(means):
    actions = tf.squeeze(means)
    actions = tf.minimum(actions, 1.0)
    actions = tf.maximum(actions, -1.0)
    return actions

def check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer):
    if t%NUM_STEPS_FOR_UPDATE==0 and len(memory_buffer)>128:
        return True
    return False


def get_experiences(memory_buffer, minibatch_size):
    df = pandas.DataFrame(memory_buffer)
    sample = df.sample(minibatch_size, replace=True)
    for key in df.columns:
        print(sample[key].values)
    return np.array(sample["states"]).astype('float32'), np.array(sample["actions"]).astype('float32'), np.array(sample["values"]).astype('float32'), np.array(sample["next_values"]).astype('float32'), np.array(sample["dones"]).astype('float32'), np.array(sample["advantages"]).astype('float32'), np.array(sample["targets"]).astype('float32'), np.array(sample["logp_ts"]).astype('float32')


def get_new_eps(epsilon):
    return max(0.9985*epsilon, 0.01)


def get_gaes(rewards, dones, values, next_values, gamma = 0.99, lamda = 0.90, normalize=True):
    deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
    deltas = np.stack(deltas)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(deltas) - 1)):
        gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

    target = gaes + values
    # gaes = np.flip(gaes)
    # target = np.flip(target)
    if normalize:
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
    return np.vstack(gaes), np.vstack(target)

def evaluate(eps, score):
    max_attempts = 1000
    env = gym.make("HalfCheetah-v4", render_mode="rgb_array")
    writer = imageio.get_writer(f'./half_cheetah/episode_{eps}_{score}.mp4', fps=30)
    state, _ = env.reset()
    tot_reward = 0
    for i in range(max_attempts):
        # state = np.asarray(state).astype('float32')
        # sin_state = np.sin(state)
        # cos_state = np.cos(state)
        # state = np.concatenate((state, sin_state, cos_state))
        state_qn = np.expand_dims(state, axis=0)
        # print(state_qn)# state needs to be the right shape for the q_network
        means = actor_network(state_qn)
        action = eval_getaction(means)
        action = tf.squeeze(action)
        env_action = action
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
    num_episodes = 10000
    batch_size = 512
    mini_batch_size = 32
    max_num_timesteps = 300
    env = gym.make("HalfCheetah-v4")
    num_p_av = 100  # number of total points to use for averaging
    epsilon = 1.0  # initial ε value for ε-greedy policy
    # Create a memory buffer D with capacity N
    state, _ = env.reset()
    i = 0
    up = 0
    score_history=[0]
    score = 0
    curr_steps = 0
    max_score = float("-inf")
    actor_network = tf.keras.models.load_model("half_cheetah_ppo_actor_max6.h5")
    critic_network = tf.keras.models.load_model("half_cheetah_ppo_critic_max6.h5")
    while i < num_episodes:
        states, next_states, actions, rewards, dones, logp_ts = [], [], [], [], [], []
        for j in range(batch_size):
            # sin_state = np.sin(state)
            # cos_state = np.cos(state)
            # state = np.concatenate((state, sin_state, cos_state))
            state_qn = np.expand_dims(state, axis=0)
            means = actor_network(state_qn)
            action, logp_t = get_action(means)
            next_state, reward, truncated, terminated, _ = env.step(action)
            curr_steps+=1
            score+=reward
            done = truncated or terminated
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            logp_ts.append(logp_t[0])
            # sin_nstate = np.sin(next_state)
            # cos_nstate = np.cos(next_state)
            # next_state_exp = np.concatenate((next_state, sin_nstate, cos_nstate))
            next_states.append(next_state)
            state = next_state.copy()
            if done:
                score_history.append(score)
                curr_steps=0
                av_latest_points=np.mean(score_history[-num_p_av:])
                if score>max_score:
                    actor_network.save('half_cheetah_ppo_actor_max6.h5')
                    critic_network.save('half_cheetah_ppo_critic_max6.h5')
                    max_score=score
                state, _ = env.reset()
                print(f"\rEpisode {i + 1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}, latest reward:{score}, max score:{max_score}",
                          end=" ")
                if (i+1)%25==0:
                    evaluate(i+1, int(av_latest_points))
                i += 1
                score = 0
        print(f"network updates, {len(states)}")
        states = np.array(states).astype("float32")
        actions = np.array(actions).astype("float32")
        rewards = np.array(rewards).astype("float32")
        dones = np.array(dones).astype("float32")
        logp_ts = np.array(logp_ts).astype("float32")
        next_states = np.array(next_states).astype("float32")
        values = critic_network(states)
        next_values = critic_network(next_states)
        advantages, targets = get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
        # experiences = [1 for s, a, v, nv, d, ad, t, l in zip(states, actions, values, next_values, dones, advantages, targets, logp_ts)]
        experiences = (states, actions, values, next_values, dones, advantages, targets, logp_ts)
        for _ in range(10):
            # mini_exp = get_experiences(experiences, mini_batch_size)
            actor_learn(experiences)
        for _ in range(10):
            # mini_exp = get_experiences(experiences, mini_batch_size)
            critic_learn(experiences)
        actor_network.save('half_cheetah_ppo_actor_5.h5')
        critic_network.save('half_cheetah_ppo_critic_5.h5')
    env.close()