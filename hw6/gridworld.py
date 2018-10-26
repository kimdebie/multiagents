#
# Multi-Agent Systems 2018
# Kim de Bie, Mathijs Pieters & Kiki van Rongen
# Homework 6: SARSA and Q-learning for Gridworld
# 26 October 2018
#

import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Specify grid size
grid_columns = 7
grid_rows = 7

# Define special cells
walls = [(1,2), (1,3), (1,4), (1,5), (2,5), (3,5), (4,5), (6,1), (6,2), (6,3)]
snakepit = (5,4)
treasure = (7,7)

# Variables for policy and updating
alpha = 0.1
discount_factor = 0.9
threshold = 0.00000000001
actions = ["N", "E", "W", "S"]
episodes = 1000

# storing performance statistics
change_per_episode_qlearning = []
change_per_episode_sarsa = []


class State():

    '''Defines a state in the grid'''

    def __init__(self, row, column, wall=False, snakepit=False, treasure=False):
        self.position = [row, column]
        self.actions = []
        self.wall = wall
        self.snakepit = snakepit
        self.treasure = treasure
        self.optimal_action = "undefined"

        if wall or snakepit or treasure:
            self.actions.append(Action(self.position,"X"))

        else:
            for action in actions:
                self.actions.append(Action(self.position, action))

    def __str__(self):
        return self.position



class Action():

    '''Defines an action that is attached to a state'''

    def __init__(self, position, direction):
        self.position = position
        self.direction = direction
        self.value = 0

    def __str__(self):
        return str(self.value)



def initialize_grid(grid_rows, grid_columns):

    '''Initialize a grid for the gridworld game, including snakepit and treasure'''

    grid = []

    for row in range(grid_rows+1):
        grid_row = []

        for column in range(grid_columns+1):

            if (row, column) in walls:
                grid_row.append(State(row, column, wall=True))
            elif (row, column) == snakepit:
                grid_row.append(State(row, column, snakepit=True))
            elif (row, column) == treasure:
                grid_row.append(State(row, column, treasure=True))
            else:
                grid_row.append(State(row, column))

        grid.append(grid_row)

    return grid



def perform_sarsa():

    '''Performs SARSA (on-policy TD) to estimate optimal state-action value'''

    time = 1

    while time < episodes:

        # select a random starting state (that is not inside a wall!)
        state = random.choice(random.choice(grid))
        while state.wall:
            state = random.choice(random.choice(grid))

        # select an initial action following an epsilon greedy policy
        action = epsilon_greedy(state, time)

        # performance statistics
        change_in_episode = 0

        # follow policy until treasure is found or snakepit is reached
        while not state.treasure and not state.snakepit:

            # follow the defined action
            moved = move(state, action)

            # observe reward and next state
            next_state = moved[0]
            reward = moved[1]

            # choose new action in new state following an epsilon-greedy policy
            next_action = epsilon_greedy(next_state, time)

            # update the Q-value of the state-action
            old_value = action.value
            action.value = action.value + alpha * (reward + discount_factor * next_action.value - action.value)

            # record performance statistics
            update_size = abs(action.value - old_value)
            change_in_episode += update_size

            # move on to the next state and action (on-policy)
            state = next_state
            action = next_action

        change_per_episode_sarsa.append(change_in_episode)
        time += 1



def perform_qlearning():

    ''' Performs Q-learning (off-policy TD) to estimate optimal state-action value'''

    time = 1

    while time < episodes:

        # select a random starting state (that is not inside a wall!)
        state = random.choice(random.choice(grid))
        while state.wall:
            state = random.choice(random.choice(grid))

        # performance statistics
        change_in_episode = 0

        while not state.treasure and not state.snakepit:

            # select action following an epsilon greedy policy
            action = epsilon_greedy(state, time)

            # follow defined action and observe reward and next state
            moved = move(state, action)
            next_state = moved[0]
            reward = moved[1]

            # define the exploitative action in the next state
            exploitative_action = exploitative_policy(next_state)

            # update Q-value of the state-action
            old_value = action.value
            action.value = action.value + alpha * (reward + discount_factor * exploitative_action.value - action.value)

            # record performance statistics
            update_size = abs(action.value - old_value)
            change_in_episode += update_size

            # move on to the next state
            state = next_state

        change_per_episode_qlearning.append(change_in_episode)
        time +=1



def move(state, action):

    '''Following an action in a state, returns consecutive state and received reward'''

    # calculate the new aimed-for position
    if action.direction == "N":
        move_to = np.add(state.position, [-1, 0])
    elif action.direction == "S":
        move_to = np.add(state.position, [1, 0])
    elif action.direction == "W":
        move_to = np.add(state.position, [0, -1])
    elif action.direction == "E":
        move_to = np.add(state.position, [0, 1])


    # we must stay within the grid, else stay in current position
    if move_to[0] > grid_rows or move_to[0] < 0 \
        or move_to[1] > grid_columns or move_to[1] < 0:

        next_state = state
        reward = -1

    else:

        considered_state = grid[move_to[0]][move_to[1]]

        if considered_state.wall:
            next_state = state
            reward = -1

        else:
            next_state = considered_state

            if considered_state.snakepit:
                reward = -10
            elif considered_state.treasure:
                reward = 20
            else:
                reward = -1

    return (next_state, reward)



def exploitative_policy(state):

    '''Find the exploitative action in a state (ie state with higest value)'''

    # initialize to arbitrarily low value
    current_high = -math.inf

    # consider all state-actions and choose action with highest value
    for action in state.actions:
        if action.value > current_high:
            current_high = action.value
            optimal_action = action

    return optimal_action



def epsilon_greedy(state, time):

    '''A GLIE version of epsilon-greedy policy selection. The probability to select
    a random action rather than a greedy one decreases over time.'''

    # follow a GLIE policy: 1 / timesteps
    pol_choice = random.uniform(0, 1)

    # below threshold choose the explorative (random) action
    if pol_choice < 1/time:
        action = random.choice(state.actions)

    # above threshold choose an exploitative (maximizing) action
    else:
        action = exploitative_policy(state)

    return action



def extract_optimal_policy():

    '''For all states, select the action with the highest Q-value as the optimal one.'''

    # consider all states in the grid
    for row in range(grid_rows+1):
        for column in range(grid_columns+1):

            state = grid[row][column]

            # consider all actions and save that with highest value
            optimal_value = -math.inf

            for action in state.actions:
                if action.value > optimal_value:
                    optimal_value = action.value
                    state.optimal_action = action.direction



def print_grid(grid):

    ''' Print the grid with its optimal policies.'''

    for row in range(grid_rows+1):
        for column in range(grid_columns+1):
            print(grid[row][column].optimal_action, end = '')
            print(" | ", end = '')
        print("")



def print_SA_values():

    '''Print optimal Q values of each state-action.'''

    for row in range(grid_rows+1):
        for column in range(grid_columns+1):

            state = grid[row][column]

            print("State at position " + str(state.position) + ":")

            for action in state.actions:
                print(action.direction + ": " + "%.2f" % action.value)

            print()



if __name__ == '__main__':
    ''' Perform both algorithms consecutively and plot their convergence speed'''

    # For SARSA:
    grid = initialize_grid(grid_rows, grid_columns)
    perform_sarsa()
    extract_optimal_policy()
    print_grid(grid)
    #print_SA_values()

    print()

    # For Q-learning:
    grid = initialize_grid(grid_rows, grid_columns)
    perform_qlearning()
    extract_optimal_policy()
    print_grid(grid)
    #print_SA_values()


    plt.plot(change_per_episode_sarsa)#, 'ro', ms=0.5)
    plt.plot(change_per_episode_qlearning)#, 'bo', ms=0.5)
    plt.show()
