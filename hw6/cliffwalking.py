#
# Multi-Agent Systems 2018
# Kim de Bie, Mathijs Pieters & Kiki van Rongen
# Homework 6: Cliffwalking
# 26 October 2018
#

import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Specify grid size
grid_columns = 9
grid_rows = 5

# Define special cells
start = (5,0)
goal = (5,9)
cliff = [(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8)]

# Variables for policy and updating
alpha = 0.1
discount_factor = 0.9
threshold = 0.00000000001
actions = ["N", "E", "W", "S"]
episodes = 1000

grid = []

class State():

    '''Defines a state in the grid'''

    def __init__(self, row, column, start=False, goal=False, cliff=False):
        self.position = [row, column]
        self.actions = []
        self.cliff = cliff
        self.start = start
        self.goal = goal
        self.optimal_action = "undefined"

        if cliff or goal:
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
        self.value = random.uniform(0, 1)

    def __str__(self):
        return str(self.value)


def initialize_grid(grid_rows, grid_columns):

    '''Initialize a grid for the cliffwalking game, including cliff and goal state'''

    grid = []

    for row in range(grid_rows+1):
        grid_row = []

        for column in range(grid_columns+1):

            if (row, column) in cliff:
                grid_row.append(State(row, column, cliff=True))
            elif (row, column) == goal:
                grid_row.append(State(row, column, goal=True))
            elif (row, column) == start:
                grid_row.append(State(row, column, start=True))
            else:
                grid_row.append(State(row, column))

        grid.append(grid_row)

    return grid


def perform_sarsa(epsilon):

    '''Performs SARSA (on-policy TD) to estimate optimal state-action value'''

    time = 1

    # performance statistics
    change_per_episode_sarsa = []
    reward_per_episode_sarsa = []

    while time < episodes:

        # select the starting state
        state = grid[start[0]][start[1]]

        # select an initial action following an epsilon greedy policy
        action = epsilon_greedy(state, epsilon)

        # performance statistics
        change_in_episode = 0

        reward_sum = 0

        # follow policy until treasure is found or snakepit is reached
        while not state.cliff and not state.goal:

            # follow the defined action
            moved = move(state, action)

            # observe reward and next state
            next_state = moved[0]
            reward = moved[1]

            reward_sum += reward

            # choose new action in new state following an epsilon-greedy policy
            next_action = epsilon_greedy(next_state, epsilon)

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
        reward_per_episode_sarsa.append(reward_sum)
        time += 1

    return(change_per_episode_sarsa, reward_per_episode_sarsa)


def perform_qlearning(epsilon):

    ''' Performs Q-learning (off-policy TD) to estimate optimal state-action value'''

    time = 1

    # performance statistics
    change_per_episode_qlearning = []
    reward_per_episode_qlearning = []

    while time < episodes:

        # select the starting state
        state = grid[start[0]][start[1]]

        # performance statistics
        change_in_episode = 0

        reward_sum = 0

        while not state.cliff and not state.goal:

            # select action following an epsilon greedy policy
            action = epsilon_greedy(state, epsilon)

            # follow defined action and observe reward and next state
            moved = move(state, action)
            next_state = moved[0]
            reward = moved[1]

            reward_sum += reward

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
        reward_per_episode_qlearning.append(reward_sum)
        time +=1

    return(change_per_episode_qlearning, reward_per_episode_qlearning)



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

        next_state = grid[move_to[0]][move_to[1]]

        if next_state.cliff:
            reward = -100
        elif next_state.goal:
            reward = 10
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



def epsilon_greedy(state, epsilon):

    '''A GLIE version of epsilon-greedy policy selection. The probability to select
    a random action rather than a greedy one decreases over time.'''

    # follow a GLIE policy: 1 / timesteps
    pol_choice = random.uniform(0, 1)

    # below threshold choose the explorative (random) action
    if pol_choice < epsilon:
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

def plot_results(results, index, num_plots, plot_rewards=True, title=None, label="run", final_plot=False):
    #change_smoothed_sarsa = pd.Series(reward_per_episode_sarsa).rolling(smoothing_window, min_periods=smoothing_window).mean()

    num = len(results)

    if plot_rewards:
        results = [results[i][1] for i in range(num)]
    else:
        results = [results[i][0] for i in range(num)]

    mean = [np.mean([results[j][i] for j in range(num)]) for i in range(episodes-1)]
    std = [np.std([results[j][i] for j in range(num)]) for i in range(episodes-1)]

    epochs = list(range(episodes-1))

    clrs = sns.color_palette("muted", num_plots)

    if title:
        plt.title(title)
        plt.xlabel("time $t$")
        if title == "Sum of rewards per episode":
            plt.ylabel("sum of rewards")
        if title == "Sum of loss per episode":
            plt.ylabel("loss")


    plt.plot(epochs, mean, label=label, c=clrs[index])
    #plt.fill_between(epochs, [mean[i]-std[i] for i in range(episodes-1)], [mean[i]+std[i] for i in range(episodes-1)], alpha=0.3, facecolor=clrs[index])

    if final_plot:
         plt.legend()
         plt.show()

def plot_grid():

    results = np.ones(shape=(grid_rows+1, grid_columns+1)) * -math.inf
    # consider all states in the grid

    for row in range(grid_rows+1):
        for column in range(grid_columns+1):
            state = grid[row][column]

            for action in state.actions:
                results[row][column] = max(action.value, results[row][column])

            if state.cliff or state.goal:
                results[row][column] = np.nan

    fig, ax = plt.subplots()
    im = ax.imshow(results)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.tight_layout()
    plt.show()

def run_experiment_sarsa(n=10, epsilon=0.05):
    global grid

    results = []
    for i in range(n):
        grid = initialize_grid(grid_rows, grid_columns)
        change, rewards = perform_sarsa(epsilon=epsilon)
        extract_optimal_policy()
        results += [[change, rewards]]

    print_grid(grid)
    print()
    return results

def run_experiment_qlearning(n=10, epsilon=0.05):
    global grid

    results = []
    for i in range(n):
        grid = initialize_grid(grid_rows, grid_columns)
        change, rewards = perform_qlearning(epsilon=epsilon)
        extract_optimal_policy()
        results += [[change, rewards]]

    print_grid(grid)
    print()
    return results


if __name__ == '__main__':
    ''' Perform both algorithms consecutively and plot their convergence speed'''

    runs = 1000

    results_sarsa1 = run_experiment_sarsa(n=runs, epsilon=0.1)
    plot_grid()
    results_qlearning1 = run_experiment_qlearning(n=runs, epsilon=0.1)
    plot_grid()

    results_sarsa05 = run_experiment_sarsa(n=runs, epsilon=0.05)
    results_qlearning05 = run_experiment_qlearning(n=runs, epsilon=0.05)

    results_sarsa01 = run_experiment_sarsa(n=runs, epsilon=0.01)
    results_qlearning01 = run_experiment_qlearning(n=runs, epsilon=0.01)

    plot_results(results_sarsa1, 0, 6, title="Sum of rewards per episode", label="SARSA $\epsilon$=0.1")
    plot_results(results_sarsa01, 1, 6, label="SARSA $\epsilon$=0.01")
    plot_results(results_sarsa05, 2, 6, label="SARSA $\epsilon$=0.05")
    plot_results(results_qlearning1, 3, 6, label="Q-learning $\epsilon$=0.1")
    plot_results(results_qlearning01, 4, 6, label="Q-learning $\epsilon$=0.01")
    plot_results(results_qlearning05, 5, 6, label="Q-learning $\epsilon$=0.05", final_plot=True)

    plot_results(results_sarsa1, 0, 6, plot_rewards=False, title="Sum of loss per episode", label="SARSA $\epsilon$=0.1")
    plot_results(results_sarsa01, 1, 6, plot_rewards=False, label="SARSA $\epsilon$=0.01")
    plot_results(results_sarsa05, 2, 6, plot_rewards=False, label="SARSA $\epsilon$=0.05")
    plot_results(results_qlearning1, 3, 6, plot_rewards=False, label="Q-learning $\epsilon$=0.1")
    plot_results(results_qlearning01, 4, 6, plot_rewards=False, label="Q-learning $\epsilon$=0.01")
    plot_results(results_qlearning05, 5, 6, plot_rewards=False, label="Q-learning $\epsilon$=0.05", final_plot=True)
