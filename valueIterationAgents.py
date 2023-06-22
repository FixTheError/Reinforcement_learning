# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #loop through actions in each state in each iteration, initializing values as the max Q for each state
        #in short, run through the Bellman equation for each iteration to initialize values
        states = self.mdp.getStates()
        for i in range(self.iterations):
            temp = self.values.copy()
            for state in states:
                Q_max = None
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    Q = self.getQValue(state, action)
                    if((Q_max == None) or (Q_max < Q)):
                        Q_max = Q
                if(Q_max == None):
                    Q_max = 0
                temp[state] = Q_max
            self.values = temp


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #use recursion to calculate a summation of transition functions * (reward + discount * value) as defined in the Q part of the Bellman equation
        Q = 0
        T_functions = self.mdp.getTransitionStatesAndProbs(state, action)
        for next, T in T_functions:
            Q += T * (self.mdp.getReward(state, action, next) + (self.discount * self.values[next]))
        return Q
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #loop through list of possible actions to find the action that produces the highest Q, if list is empty return none.
        actions = self.mdp.getPossibleActions(state)
        V = 0
        computed = None
        if(len(actions) == 0):
            return computed
        for action in actions:
            temp = self.getQValue(state, action)
            if((temp > V) or (computed == None)):
                V = temp
                computed = action
        return computed
        util.raiseNotDefined()
        
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
