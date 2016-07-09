import random
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class QLearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    

    def __init__(self, env):
        super(QLearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Q = {}
        self.learning_rate = 0.9
        self.states = None
        self.possible_actions = [None, 'forward', 'left', 'right']
        

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)



        # TODO: Update state
        # state = (self.next_waypoint,inputs['light'], inputs['oncoming'], inputs['right'], inputs['left'],deadline)
        state = (self.next_waypoint,inputs['light'], inputs['oncoming'], inputs['right'], inputs['left'])
        # state = (self.next_waypoint,inputs['light'])
        self.state = state

        for action in self.possible_actions:
            if(state, action) not in self.Q: 
                self.Q[(state, action)] = 20

        # TODO: Select action according to your policy
        Q_best_value = [self.Q[(state, None)], self.Q[(state, 'forward')], self.Q[(state, 'left')], self.Q[(state, 'right')]]
        # Softmax
        Q_best_value = np.exp(Q_best_value) / np.sum(np.exp(Q_best_value), axis=0)
        
        #action = np.random.choice(self.possible_actions, p=probabilities)
        action = self.possible_actions[np.argmax(Q_best_value)]

        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward == -10.0 :
            print "Invalid move"
        if reward == -0.5 :
            print "valid, but is it correct?"
        

        # TODO: Learn policy based on state, action, reward
        # self.Q[(state,action)] = reward + self.learning_rate * self.Q[(state,action)]
        self.Q[(state,action)] = reward * self.learning_rate + ( 1 - self.learning_rate ) * self.Q[(state,action)]
        # print self.Q[(state, action)]


        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]



class RandomAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(RandomAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)



        # TODO: Update state
        self.state = (
            self.next_waypoint,
            inputs['light'],
            inputs['oncoming'],
            inputs['right'],
            inputs['left'],
            deadline)
        
        
        # TODO: Select action according to your policy
        possible_action_states = [None, 'forward', 'left', 'right']
        action = possible_action_states[random.randint(0, 3)]

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward


        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    # a = e.create_agent(RandomAgent)  # create agent
    a = e.create_agent(QLearningAgent)  # create agent
    # e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.01, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
