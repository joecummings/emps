import sys
import numpy as np
import math
import random

import gym
import gym_maze
from gym_maze.envs.maze_view_2d import Maze, MazeView2D
from gym_maze.envs.maze_env import max_maze_file_digit


curr_state = np.asarray([0.5, 0.5, 0.5, 0.5])  #[happy, sad, bored, frustrated]
mood = np.asarray([0.5, 0.5, 0.5, 0.5])
curr_emotion = curr_state.index(max(curr_state))

def simulate():

    # Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_rate = get_discount_rate(0)

    num_streaks = 0

    # Render tha maze
    env.render()

    prev_rewards = []

    for episode in range(NUM_EPISODES):

        # Reset the environment
        obv = env.reset()

        # the initial state
        state_0 = state_to_bucket(obv)
        total_reward = 0

        for t in range(MAX_T):

            # Update parameters
            explore_rate = get_explore_rate(t)
            learning_rate = get_learning_rate(t)
            # discount_rate += get_discount_rate(t)

            # Select an action
            action = select_action(state_0, explore_rate)
            
            # execute the action
            obv, reward, done, info = env.step(action)

            # Observe the result
            state = state_to_bucket(obv)
            total_reward += reward
            prev_rewards.insert(0, reward)

            # update emotion state
            if curr_state.index(max(curr_state)) != curr_emotion:
                update_emotion_state(t, prev_rewards)

            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate * (reward + discount_rate * (best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state

            # Print data
            if DEBUG_MODE > 1:
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Total Reward: %f" % total_reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Discount rate:", discount_rate)
                print("Q: ", q_table)
                print("Streaks: %d" % num_streaks)
                print("Total reward: %f" % total_reward)
                print("")

            if DEBUG_MODE > 0:
                if done or t >= MAX_T - 1:
                    print("\nEpisode = %d" % episode)
                    print("t = %d" % t)
                    print("Explore rate: %f" % explore_rate)
                    print("Learning rate: %f" % learning_rate)
                    print("Streaks: %d" % num_streaks)
                    print("Total reward: %f" % total_reward)
                    print("")

            # Render tha maze
            if RENDER_MAZE:
                env.render()

            if env.is_game_over():
                return
                # sys.exit()

            if done:
                print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                      % (episode, t, total_reward, num_streaks))

                if t <= SOLVED_T:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

            elif t >= MAX_T - 1:
                print("Episode %d timed out at %d with total reward = %f."
                      % (episode, t, total_reward))

        if DEBUG_MODE > 2:
            wait = input("Press enter to continue to next episode...")

        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > STREAK_TO_END or True:
            break


def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = int(np.argmax(q_table[state]))
    return action

def update_emotion_state(t, prev_reward):
    delta = 0.0
    
    try:
        delta = abs(prev_reward[0]/prev_reward[1])
        
        if prev_reward[1] > prev_reward[0]:
            delta = delta * -1.0
        
    except:
       delta = 0.0

    if delta > 2.0:
        curr_state[0] = min(curr_state[0] + 0.1, 1.0) #happiness goes up
        curr_state[1] = max(curr_state[1] - 0.1, 0.0) #sadness goes down
        curr_state[2] = max(curr_state[2] - 0.1, 0.0) #boredom goes down
        curr_state[3] = max(curr_state[3] - 0.1, 0.0) #frustration goes down
    
    elif delta < 2.0:
        curr_state[0] =  max(curr_state[0] - 0.1, 0.0) #happiness goes down
        curr_state[1] = min(curr_state[1] + 0.1, 1.0) #sadness goes up
        curr_state[2] = max(curr_state[2] - 0.1, 0.0) #boredom goes down
    
    else:

    



def get_explore_rate(er):
    const_emo_er = [-0.20, -0.10, 0.10, 0.10]

    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - np.dot(const_emo_er, curr_state)))


def get_learning_rate(t):
    const_emo_lr = [0.2, -0.1, -0.1, 0.1]

    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - np.dot(const_emo_lr, curr_state)))


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


if __name__ == '__main__':
    """Run the test
    """

    max_maze = int(input("Max number of mazes to test: "))
    gym_maze.envs.maze_env.max_maze_file_digit = max_maze
    curr_maze = 0
    while curr_maze < max_maze:
        curr_maze = curr_maze + 1

        # Initialize the "maze" environment
        env = gym.make("maze-5x5-reward-v0")

        '''
        Defining the environment related constants
        '''
        # Number of discrete states (bucket) per state dimension
        MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
        NUM_BUCKETS = MAZE_SIZE  # one bucket per grid

        # Number of discrete actions
        NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
        # Bounds for each discrete state
        STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

        '''
        Learning related constants
        '''
        MIN_EXPLORE_RATE = 0.001
        MIN_LEARNING_RATE = 0.2
        DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0

        '''
        Defining the simulation related constants
        '''
        NUM_EPISODES = 50000
        MAX_T = np.prod(MAZE_SIZE, dtype=int) * 100
        STREAK_TO_END = 100
        SOLVED_T = np.prod(MAZE_SIZE, dtype=int)
        DEBUG_MODE = 2  # [0...3]; less <---> more verbose
        RENDER_MAZE = True
        ENABLE_RECORDING = True

        '''
        Creating a Q-Table for each state-action pair
        '''
        q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

        '''
        Begin simulation
        '''
        recording_folder = "/tmp/maze_q_learning"

        # if ENABLE_RECORDING:
        #    env.monitor.start(recording_folder, force=True)

        simulate()

        # if ENABLE_RECORDING:
        #    env.monitor.close()
