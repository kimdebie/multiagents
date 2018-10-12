# De recursies lopen lelijk naar elkaar door
# maakt het uit in welke volgorde policies worden bekeken? lijkt nu wel zo...


import numpy as np
import random

grid_columns = 4
grid_rows = 4

discount_factor = 0.9
threshold = 0.00000000001
policy = ["N", "E", "W", "S"]

def initialize_grid(grid_rows, grid_columns):
    grid = []

    for row in range(grid_rows+1):
        grid_row = []

        for column in range(grid_columns+1):
            grid_row.append(State(row, column))


        grid.append(grid_row)

    return grid

class State():
    '''Defines a state in the grid'''

    def __init__(self, row, column):
        self.position = [row, column]
        self.value = 0
        self.policy = "random"

    def __str__(self):
        return str(self.value)

def determine_possible_states(action, state_considered):

    # The special moves

    if state_considered.position == [0, 1]:
        new_state = grid[4][1]
        reward = 10
        return [reward, new_state]
    elif state_considered.position == [0, 3]:
        new_state = grid[2][3]
        reward = 5
        return [reward, new_state]

    if action == "N":
        move_to = np.add(state_considered.position, [-1, 0])
    elif action == "S":
        move_to = np.add(state_considered.position, [1, 0])
    elif action == "W":
        move_to = np.add(state_considered.position, [0, -1])
    elif action == "E":
        move_to = np.add(state_considered.position, [0, 1])


    # Result of the moves

    if move_to[0] <= 4 and move_to[0] >= 0 and move_to[1] <= 4 and move_to[1] >= 0: # ordinary move
        new_state = grid[move_to[0]][move_to[1]]
        reward = 0
    else: # move impossible
        new_state = state_considered
        reward = -1

    return [reward, new_state]

def evaluate_policy():

    # Following pseudocode of Barto & Sutton p. 63

    round = 0
    while True:

        change = 0

        # look at each state
        for row in range(grid_rows+1):

            for column in range(grid_columns+1):

                state_considered = grid[row][column]

                old_value = state_considered.value
                updated_value = 0

                # look at all possible actions
                if state_considered.policy == "random":
                    possible_policies = policy
                else:
                    possible_policies = state_considered.policy

                for action in possible_policies:

                    outcomes_of_action = determine_possible_states(action, state_considered)

                    reward = outcomes_of_action[0]
                    value_new_state = outcomes_of_action[1].value

                    updated_value += 0.25 * (reward + discount_factor * value_new_state)

                change = max(change, np.abs(updated_value - old_value)) # however value function changed

                state_considered.value = updated_value

        if change < threshold:
            print("Evaluation completed; moving to greedification")
            greedify()
            break

def greedify():

    # Following pseudocode of Sutton & Barto p. 65

    policy_stable = True

    for row in range(grid_rows+1):
        for column in range(grid_columns+1):

            state_considered = grid[row][column]

            old_policy = state_considered.policy

            value_of_action = -1000000000000

            np.random.shuffle(policy) # dit breekt het hele ding, waarom?

            for action in policy:

                outcomes_of_action = determine_possible_states(action, state_considered)

                reward = outcomes_of_action[0]

                if reward > value_of_action:
                    value_of_action = reward
                    new_policy = action

            state_considered.policy = new_policy

            if new_policy != old_policy:
                policy_stable = False

    if policy_stable:
        print("The policy has stabilized")
        print_grid(grid)

    else:
        evaluate_policy()



def print_grid(grid):
    for row in range(grid_rows+1):
        for column in range(grid_columns+1):
            print("%.2f" % grid[row][column].value, end = '')
            print(" | ", end = '')
        print("")


def print_grid_policies(grid):
    for row in range(grid_rows+1):
        for column in range(grid_columns+1):
            print(grid[row][column].policy, end = '')
            print(" | ", end = '')
        print("")







grid = initialize_grid(grid_rows, grid_columns)
evaluate_policy()
print_grid(grid)
print_grid_policies(grid)
