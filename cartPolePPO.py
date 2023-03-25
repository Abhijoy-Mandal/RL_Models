import numpy as np
import tensorflow as tf
import gym
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls
env= gym.make("CartPole-v0")
low = env.observation_space.low
high = env.observation_space.high

class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.v = tf.keras.layers.Dense(1, activation=None)

    def call(self, input_data):
        x = self.d1(input_data)
        v = self.v(x)
        return v


class Actor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.a = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, input_data):
        x = self.d1(input_data)
        a = self.a(x)
        return a


class Agent():
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        # self.a_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        # self.c_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.actor = Actor()
        self.critic = Critic()
        self.clip_pram = 0.2

    def act(self, state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def actor_loss(self, probs, actions, adv, old_probs, closs):
        probability = probs
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability, tf.math.log(probability))))
        # print(probability)
        # print(entropy)
        sur1 = []
        sur2 = []

        for pb, t, op, a in zip(probability, adv, old_probs, actions):
            t = tf.constant(t)
            # op =  tf.constant(op)
            # print(f"t{t}")
            # ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
            ratio = tf.math.divide(pb[a], op[a])
            # print(f"ratio{ratio}")
            s1 = tf.math.multiply(ratio, t)
            # print(f"s1{s1}")
            s2 = tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram), t)
            # print(f"s2{s2}")
            sur1.append(s1)
            sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)

        # closs = tf.reduce_mean(tf.math.square(td))
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        # print(loss)
        return loss

    def learn(self, states, actions, adv, old_probs, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        adv = tf.reshape(adv, (len(adv),))

        old_p = old_probs

        old_p = tf.reshape(old_p, (len(old_p), 2))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v = self.critic(states, training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            c_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss)

        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss


def preprocess1(states, actions, rewards, done, values, gamma):
    g = 0
    lmbda = 0.95
    returns = []
    for i in reversed(range(len(rewards))):
       delta = rewards[i] + gamma * values[i + 1] * done[i] - values[i]
       g = delta + gamma * lmbda * dones[i] * g
       returns.append(g + values[i])

    returns.reverse()
    adv = np.array(returns, dtype=np.float32) - values[:-1]
    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    returns = np.array(returns, dtype=np.float32)
    return states, actions, returns, adv

def test_reward(env):
  total_reward = 0
  state = env.reset()
  done = False
  while not done:
    action = np.argmax(agent.actor(np.array([state])).numpy())
    next_state, reward, done, _ = env.step(action)
    state = next_state
    total_reward += reward

  return total_reward


if __name__=="__main__":
    agent = Agent()
    max_episodes = 5000
    max_num_timesteps = 500
    num_p_av = 100
    ep_reward = []
    target = False
    best_reward = 0
    avg_rewards_list = []
    total_point_history = []

    for i in range(max_episodes):
        if target == True:
            break

        done = False
        state = env.reset()
        all_aloss = []
        all_closs = []
        rewards = []
        states = []
        actions = []
        probs = []
        dones = []
        values = []
        episode_reward = 0
        for j in range(max_num_timesteps):

            action = agent.act(state)
            value = agent.critic(np.array([state])).numpy()
            next_state, reward, done, _ = env.step(action)
            episode_reward+=reward
            dones.append(1 - done)
            rewards.append(reward)
            states.append(state)
            # actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())
            actions.append(action)
            prob = agent.actor(np.array([state]))
            probs.append(prob[0])
            values.append(value[0][0])
            state = next_state
            if done:
                env.reset()

        value = agent.critic(np.array([state])).numpy()
        values.append(value[0][0])
        np.reshape(probs, (len(probs), 2))
        probs = np.stack(probs, axis=0)
        total_point_history.append(episode_reward)

        states, actions, returns, adv = preprocess1(states, actions, rewards, dones, values, 1)

        for epocs in range(10):
            al, cl = agent.learn(states, actions, adv, probs, returns)
            # print(f"al{al}")
            # print(f"cl{cl}")
        av_latest_points = np.mean(total_point_history[-num_p_av:])
        print(f"\rEpisode {i + 1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}, latest reward:{episode_reward}",
              end="")
        if (i+1)%num_p_av==0:
            agent.actor.save('cartpole_model_actor_{}_{}'.format(i, av_latest_points), save_format="tf")
            agent.critic.save('cartpole_model_critic_{}_{}'.format(i, av_latest_points), save_format="tf")
        if episode_reward == 200:
            agent.actor.save('cartpole_model_actor_{}_{}'.format(i, av_latest_points), save_format="tf")
            agent.critic.save('cartpole_model_critic_{}_{}'.format(i, av_latest_points), save_format="tf")
            target = True
        env.reset()

    env.close()