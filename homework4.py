import random
import numpy
import math
import pandas as pd
import matplotlib.pyplot as plt


methods = ["epsilon-greedy", "initial-optimistic", "ucb"]
method = "ucb"

arms = 20
runs = 1000
bandits = []
rounds = 300

#data = pd.DataFrame(index=range(1, runs+1), columns=methods)
#data = data.fillna(0.0)

epsilons = [0.1, 0.5, 0.9]
data = pd.DataFrame(index=range(1, runs+1), columns=epsilons)
data = data.fillna(0.0)

# Parameters (adopted values from slides)
epsilon = 0.1 # for e-greedy
c = 2 # for UCB

class BanditArm(object):
    '''Bandit arm'''

    def __init__(self, number):
        self.numberOfArm = number
        self.trueMean = random.uniform(0, 100)
        self.sd = 1#random.uniform(1,5)
        self.chosen = 0

    def setInitialMean(self, method):
        if method == "epsilon-greedy":
            self.beliefMean = 0
        elif method == "initial-optimistic":
            self.beliefMean = 150
        elif method == "ucb":
            self.beliefMean = 0
        else:
            print("No valid selection mechanism found")

def initialize(method):
    '''Initialize arms and store in array'''

    del bandits[:]

    for arm in range(arms):
        bandit = BanditArm(arm)
        bandit.setInitialMean(method)
        bandits.append(bandit)

def run(method):

    for run in range(1, runs+1):

        # select action
        if method == "initial-optimistic":
            '''Action is fully greedy: the highest beliefMean is selected'''
            beliefMean = 0
            for bandit in bandits:
                if bandit.beliefMean > beliefMean:
                    selectedBandit = bandit
                    beliefMean = bandit.beliefMean

        elif method == "epsilon-greedy":
            '''A random arm is selected with probability epsilon; the greedy action is executed otherwise'''
            prob = random.uniform(0,1)
            if prob <= epsilon:
                selectedBandit = bandits[random.randint(0, arms-1)]
            else:
                beliefMean = -1
                for bandit in bandits:
                    if bandit.beliefMean > beliefMean:
                        selectedBandit = bandit
                        beliefMean = bandit.beliefMean

        elif method == "ucb":
            '''The arm with the highest upper-bound confidence value is selected'''
            maxUpperBound = -1
            for bandit in bandits:
                if bandit.chosen == 0:
                    upperBound = 1e400 # upper bound set to infinitely large value (so that bandit is chosen)
                else:
                    upperBound = bandit.beliefMean + c * math.sqrt(math.log(run)/(2*bandit.chosen)) # formula from slides

                if upperBound > maxUpperBound:
                    selectedBandit = bandit
                    maxUpperBound = upperBound

        else:
            print("Missing method")
            break

        selectedBandit.chosen += 1
        reward = numpy.random.normal(selectedBandit.trueMean, selectedBandit.sd)

        # update the average reward at this run (over all rounds) in the dataset
        #data.at[run, method] = (data.at[run, method] * round + reward) / (round + 1)
        data.at[run, epsilon] = (data.at[run, epsilon] * round + reward) / (round + 1)


        # updating belief status (formula from slides)
        selectedBandit.beliefMean = selectedBandit.beliefMean + 1 / (selectedBandit.chosen + 1) * (reward - selectedBandit.beliefMean)


def visualize_all():
    ax = plt.gca()

    data.plot(kind='line', y='epsilon-greedy', color='blue',ax=ax, title="k-armed Bandit using 3 Forms of Exploration/Exploitation")
    data.plot(kind='line', y='initial-optimistic', color='red', ax=ax)
    data.plot(kind='line', y='ucb', color='green', ax=ax)

    plt.show()

def visualize_egreedy():
    ax = plt.gca()

    data.plot(kind='line', y=0.1, color='#d9f0a3',ax=ax, title="k-armed Bandit using the epsilon-greedy method")
    #data.plot(kind='line', y=0.2, color='#f7fcb9', ax=ax)
    #data.plot(kind='line', y=0.3, color='#d9f0a3', ax=ax)
    #data.plot(kind='line', y=0.4, color='#addd8e', ax=ax)
    data.plot(kind='line', y=0.5, color='#78c679', ax=ax)
    #data.plot(kind='line', y=0.6, color='#41ab5d', ax=ax)
    #data.plot(kind='line', y=0.7, color='#238443', ax=ax)
    #data.plot(kind='line', y=0.8, color='#006837', ax=ax)
    data.plot(kind='line', y=0.9, color='#004529', ax=ax)
    #data.plot(kind='line', y=1.0, color='black', ax=ax)

    plt.show()


if __name__ == '__main__':

    # for method in methods:
    #     print(method)
    #     for round in range(rounds):
    #         initialize(method)
    #         run(method)
    #
    # visualize_all()


    for eps in epsilons:
        epsilon = eps
        for round in range(rounds):
            initialize("epsilon-greedy")
            run("epsilon-greedy")

    visualize_egreedy()
