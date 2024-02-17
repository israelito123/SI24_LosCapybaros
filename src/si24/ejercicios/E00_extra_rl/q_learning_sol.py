import gymnasium as gym
import numpy as np


class RandomAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.num_actions = env.action_space.n

        # Tabla estados x acciones
        self.Q = np.zeros((env.observation_space.n,
                           env.action_space.n))
        # Parameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def act(self, observation):
        return self.action_space.sample()

    def step(self, state, action, reward, next_state):
        pass


class QLearningAgent(RandomAgent):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(env, alpha, gamma, epsilon)

    def save(self, filename):
        np.save(filename, self.Q)

    def act(self, observation):
        if np.random.random() < self.epsilon:
            return env.action_space.sample()  # Exploration
        else:
            return np.argmax(self.Q[observation])  # Exploitation

    # Update Q values using Q-learning
    def step(self, state, action, reward, next_state):
        # td_target = reward + self.gamma * np.max(self.Q[next_state])
        # td_error = td_target - self.Q[state][action]
        # self.Q[state][action] += self.alpha * td_error

        self.Q[state][action] = self.Q[state][action] + \
            self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])


if __name__ == "__main__":
    # https://gymnasium.farama.org/environments/toy_text/cliff_walking/
    env = gym.make("CliffWalking-v0")

    n_episodes = 1000
    episode_length = 200
    agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.9)
    for e in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0
        for i in range(episode_length):
            # take a random action
            action = agent.act(obs)
            next_obs, reward, done, _, _ = env.step(action)
            # update agent
            agent.step(obs, action, reward, next_obs)

            if done:
                break
            ep_return += reward
            obs = next_obs
            print(agent.Q)
        
        # Decay epsilon
        agent.epsilon *= 0.99
        print(f"Episode {e} return: ", ep_return)
    env.close()
