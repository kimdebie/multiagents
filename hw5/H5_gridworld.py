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

def determine_possible_states(action, state_considered, gametype):

    if action == "N":
        move_to = np.add(state_considered.position, [-1, 0])
    elif action == "S":
        move_to = np.add(state_considered.position, [1, 0])
    elif action == "W":
        move_to = np.add(state_considered.position, [0, -1])
    elif action == "E":
        move_to = np.add(state_considered.position, [0, 1])

    # The special moves in infinite vs episodic games
    if gametype == 'infinite':
        if state_considered.position == [0, 1]:
            new_state = grid[4][1]
            reward = 10
            return [reward, new_state]
        elif state_considered.position == [0, 3]:
            new_state = grid[2][3]
            reward = 5
            return [reward, new_state]

    elif gametype == 'episodic':
        if state_considered.position == [0, 1]:
            new_state = grid[0][1]
            reward = 0
            return [reward, new_state]
        elif state_considered.position == [0, 3]:
            new_state = grid[0][3]
            reward = 0
            return [reward, new_state]
        elif move_to[0] == 0 and move_to[1] == 1:
            new_state = grid[0][1]
            reward = 10
            return [reward, new_state]
        elif move_to[0] == 0 and move_to[1] == 3:
            new_state = grid[0][3]
            reward = 5
            return [reward, new_state]

    # Regular/off-the-grid moves

    if move_to[0] <= 4 and move_to[0] >= 0 and move_to[1] <= 4 and move_to[1] >= 0: # ordinary move
        new_state = grid[move_to[0]][move_to[1]]
        reward = 0
    else: # move impossible
        new_state = state_considered
        reward = -1

    return [reward, new_state]

def evaluate_policy(gametype='infinite'):

    if gametype == 'episodic':
        global discount_factor
        discount_factor = 1.0

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

                    outcomes_of_action = determine_possible_states(action, state_considered, gametype)

                    reward = outcomes_of_action[0]
                    value_new_state = outcomes_of_action[1].value

                    updated_value += 0.25 * (reward + discount_factor * value_new_state)

                change = max(change, np.abs(updated_value - old_value)) # however value function changed

                state_considered.value = updated_value

        if change < threshold:
            break

def policy_update(action, state_considered, gametype):

    outcomes_of_action = determine_possible_states(action, state_considered, gametype)
    reward_current_action = outcomes_of_action[0]
    value_new_state = outcomes_of_action[1].value

    value_of_action = reward_current_action + discount_factor * value_new_state

    return value_of_action

def greedify(gametype='infinite'):

    if gametype == 'episodic':
        global discount_factor
        discount_factor = 1.0

    while True:
        evaluate_policy(gametype)

        # Following pseudocode of Sutton & Barto p. 65

        policy_stable = True

        # consider each state
        for row in range(grid_rows+1):

            for column in range(grid_columns+1):

                state_considered = grid[row][column]

                # the action taken using the current policy
                old_policy = state_considered.policy

                # the value of this action
                if old_policy == "random":
                    value_of_action = -1000000000000
                else: # look ahead one step
                    value_of_action = state_considered.value


                new_policy = old_policy

                for action in policy:

                    reward = policy_update(action, state_considered, gametype)

                    if reward > value_of_action:
                        value_of_action = reward
                        new_policy = action

                state_considered.policy = new_policy

                if new_policy != old_policy:
                    policy_stable = False

        if policy_stable:
            break


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

def main():

    # Policy evaluation for infinite game
    print("Policy evaluation of a random probabilistic policy in infinity")
    evaluate_policy()
    print_grid(grid)

    # Policy iteration
    print("Turning the policy into a greedy one in infinity")
    greedify()
    print_grid(grid)
    print_grid_policies(grid)

    # Policy evaluation for episodic game
    print("Policy evaluation of the random probabilistic policy in the episodic version of the game")
    evaluate_policy('episodic')
    print_grid(grid)

    # Policy iteration episodic
    print("Policy greedification for the episodic game")
    greedify('episodic')
    print_grid(grid)
    print_grid_policies(grid)


if __name__ == '__main__':
    main()
