import random
import numpy as np

class MctsNode():
    def __init__(self, state, actions=None, parent=None):
        self.visits = 0
        self.reward = 0.0
        self.state = state
        self.parent = parent
        self.children = []
        self.action_to_child = []
        self.untried_actions = actions if actions else state.actions()
    
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def rollout(self):
        state = self.state
        while not state.terminal_test():
            action = random.choice(state.actions())
            state = state.result(action)
        return -1 if state._has_liberties(self.state.player()) else 1
    
    def backpropagate(self, reward):
        node = self
        while node:
            node.reward += reward
            node.visits += 1
            node = node.parent
            reward *= -1
    
    def best_child(self, c_param=1.0):
        visits = self.visits
        choices_weights = [(c.reward / (c.visits)) + c_param * np.sqrt((2 * np.log(visits) / (c.visits))) for c in self.children]
        return self.children[np.argmax(choices_weights)]
    
    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.result(action)
        next_state_action = next_state.actions()
        child_node = MctsNode(next_state, next_state_action, self)
        self.children.append(child_node)
        self.action_to_child.append(action)
        return child_node
    
    def best_child_action(self, child):
        return self.action_to_child[self.children.index(child)]
