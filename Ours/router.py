"""
Created on Fri May 19 16:07:55 2023

@author: YangYing
"""
import copy

class router:
    def __init__(self, K_paths):
        self.K_paths = K_paths
        self.info = []
        self.state = []
        self.actions = []
        self.max_hop = 0 
        self.rate_info = []
        self.local_rewards = []
        self.total_state = []
        self.total_rewards = []
        
    def reset(self):
        self.info = []
        self.state = []
        self.FedState = []
        self.actions = []
        self.max_hop = 0
        self.rate_info = []
        self.local_rewards = []
        self.total_state = []
        self.total_rewards = []
        
    def add_info(self, index, pathNum):
        self.info.append([index, pathNum])
    
    def add_state(self, part_state):
        self.state.append(part_state)
    
    def add_total_state(self, total_state):
        self.total_state.append(total_state)

    def add_FedState(self, state):
        new_state = copy.deepcopy(self.state[0])
        for i in range(len(state[0])):
            new_state.append(state[0][i])
        self.FedState.append(new_state)

    def add_action(self, actions):
        for i in range(len(actions)):
            for j in range(self.K_paths - self.info[i][1]):
                actions[i][self.K_paths-1-j] = 0
            actionSum = sum(actions[i])
            if actionSum == 0:
                actions[i][0] = 1
            else:
                for j in range(len(actions[i])):
                    actions[i][j] /= actionSum
                    actions[i][j] = round(actions[i][j], 2)
        self.actions = actions
            
    def add_rateInfo(self, index, rate):
        self.rate_info.append([index, rate])    
    
    def add_local_reward(self, reward):
        self.local_rewards.append(reward)

    def add_total_reward(self, total_reward):
        self.total_rewards.append(total_reward)
    
    
    def getInfo(self):
        return self.info
    
    def getLocalState(self):
        return self.state
    
    def getState(self):
        return self.total_state

    def getFedState(self):
        return self.FedState

    def getAction(self):
        return self.actions
    
    def getRateInfo(self):
        return self.rate_info
    
    def getLocalRewards(self):
        return self.local_rewards
    
    def getRewards(self):
        return self.total_rewards

    def getRewards_load(self):
        return self.load