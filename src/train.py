import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle




def run(episodes, is_training=True,render = False):

    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human" if render else None) # create the environment    
    
    if (is_training):
        q_table = np.zeros((env.observation_space.n, env.action_space.n)) # initialize 16 * 4 array
    else:
        # f = open("frozen_lake.pkl", "rb")
        # q_table=pickle.load(f)
        # f.close()
        with open("frozen_lake.pkl", "rb") as f:
            q_table = pickle.load(f)

    learning_rate = 0.9 # alpha or learning_rate
    discount_factor = 0.9 # gamma or discount_factor

    epsilon = 1  # 1 = 100% random action
    epsilon_decay = 0.0001  # epsilon decay rate 1/0.0001 = 10000
    rng = np.random.default_rng() # create a random number generator

    reward_per_episode = np.zeros(episodes) # initialize an array to store rewards for each episode 
     
    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        


        while not terminated and not truncated:
            if is_training and rng.random() < epsilon: # explore
                action = env.action_space.sample()  # actions left:0, down:1, right:2, up:3
            else: # exploit
                action = np.argmax(q_table[state,:]) # choose the action with the highest Q-value for the current state

            new_state, reward, terminated, truncated, _ = env.step(action)
            reward_per_episode[i] += reward

            if is_training:
                q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state,:]) - q_table[state, action])



            state = new_state

        epsilon = max(0, epsilon - epsilon_decay) # decay epsilon

    # if (epsilon==0):
    #     learning_rate = 0.0001

    env.close()


    sum_rewards = np.zeros(episodes)

    for t in range(episodes):
        sum_rewards[t] = np.sum(reward_per_episode[max(0, t-100): (t+1)]) # calculate the cumulative reward for each episode
    
    plt.plot(sum_rewards)
    plt.savefig("frozen_lake.png")
    
    if is_training:
        f = open("frozen_lake.pkl", "wb")
        pickle.dump(q_table,f)
        f.close()


if __name__ == "__main__":
    run(1000, is_training=False, render=True)