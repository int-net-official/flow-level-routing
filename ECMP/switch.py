"""
Created on Fri May 19 16:07:55 2023

@author: YangYing
"""
class switch:
    def __init__(self, splitNum):
        self.reminderPath = []
        self.InRate = []
        self.OutRate = []
        self.splitNum = splitNum
    
    def reset(self):
        self.reminderPath = []
        self.InRate = []
        self.OutRate = []
    
    def addReminderPath(self, flowIndex, splitIndex, reminderPath):
        index = flowIndex * self.splitNum + splitIndex
        self.reminderPath.append([index, reminderPath])
    
    def addInRate(self, flowIndex, splitIndex, InRate):
        add = True
        index = flowIndex * self.splitNum + splitIndex
        for i in range(len(self.InRate)):
            if index == self.InRate[i][0]:
                self.InRate[i][1] = InRate
                add = False
                break
        if add:
            self.InRate.append([index, InRate])
            
    def addInRate2(self, index, InRate):
        add = True
        for i in range(len(self.InRate)):
            if index == self.InRate[i][0]:
                self.InRate[i][1] = InRate
                add = False
                break
        if add:
            self.InRate.append([index, InRate])
    
    def addOutRate(self, flowIndex, splitIndex, OutRate):
        index = flowIndex * self.splitNum + splitIndex
        self.OutRate.append([index, OutRate])
   
    def addOutRate2(self, index, OutRate):
        self.OutRate.append([index, OutRate])    

    def getSumInrate(self):
        ans = 0
        for i in range(len(self.InRate)):
            ans += self.InRate[i][1]
        return ans
    
    def getOutRate(self, flowIndex, splitIndex):
        index = flowIndex * self.splitNum + splitIndex
        for i in range(len(self.OutRate)):
            if (self.OutRate[i][0] == index):
                return self.OutRate[i][1]

    def addEdgeRate(self, pathNode, next_pathNode, rate):
        self.G.edges[(pathNode, next_pathNode)]['load'] += rate