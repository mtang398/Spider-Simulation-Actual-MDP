from statistics import STATISTICS
from utils import Random, Infinity, LargeInteger
from simulator import Status, PHASE
import numpy as np
from game import Game, GameState
class Value:
    def __init__(self):
        self.Count = 0.0
        self.Total = 0.0

    def Set(self, count, value):
        self.Count = float(count)
        self.Total = float(value) * float(count)

    def Add(self, totalReward, weight=1):
        self.Count += 1.0
        self.Total += float(totalReward) * float(weight)

    def GetValue(self):
        if self.Count == 0:
            return self.Total
        else:
            return float(self.Total) / float(self.Count)

    def GetTrueValue(self):
        return float(self.Total)

    def GetCount(self):
        return self.Count

    def __str__(self):
        return "(" + str(self.Count) + " , " + str(self.Total) + ")"

class Node:

    def __init__(self, state, simulator, parent = None):
        self.GameState = state
        self.Children = [] #children of vnodes
        self.Value = Value()
        self.parent = parent
        self.Simulator = simulator
        #self.Status = Status()

    def Child(self, c):
        return self.Children[c]

    def GetNumChildren(self):
        return len(self.Children)
    
    #-------------------------------------------------
    def selection(self, state):
        if self.Children == []:
            return self
        else:
            index = self.SelectAction(state)
            #print(index)
            #print(self.Child(index).GameState.AgentPos.X)
            #print(self.Child(index).GameState.AgentPos.Y)
            return self.Child(index).selection(self.Child(index).GameState)
    
    def Simulation(self, state):
        totalReward = 0.0
        discount = 1.0
        terminal = False
        numSteps = 0

        currentState = state.copy()
        while numSteps < 100 and not terminal:
            action = Random(0, self.Simulator.NumActions)
            terminal, state, reward = self.Simulator.Step(currentState, action)

            totalReward += reward*discount
            discount *= self.Simulator.GetDiscount()
            numSteps += 1

        return totalReward

    def ExpandNode(self, state):
        for action in range(self.Simulator.NumActions):
            temp = state.copy()
            terminal, next_state, reward = self.Simulator.Step(temp,action)
            #print(state.AgentPos.X)
            #print(state.AgentPos.Y)
            if reward < -400:
                self.Children.append(None)
            else:
                childNode = Node(next_state, self.Simulator, parent = self)
                self.Children.append(childNode)
        return self
    
    def BackPropagation(self, reward):
        self.Value.Add(reward)
        if self.parent:
            self.parent.BackPropagation(reward)
            
#------------------------------------------------------
    def FastUCB(self, N, n, logN):
        if n == 0:
            return Infinity
        else:
            return np.sqrt(2) * np.sqrt(logN / n)
        
    def SelectAction(self,state):
        N = self.Value.GetCount()
        logN = np.log(N+1)
        
        UCB_Values = []
        for action in range(self.Simulator.NumActions):
                
            ChildNode = self.Child(action)
            if ChildNode:
                #print(ChildNode.GameState.AgentPos.X)
                #print(ChildNode.GameState.AgentPos.Y)
                q = ChildNode.Value.GetValue()
                n = ChildNode.Value.GetCount()
                if n != 0:
                    ucb = q/n + self.FastUCB(N,n,logN)
                else:
                    ucb = self.FastUCB(N,n,logN)
                UCB_Values.append(ucb)
            elif ChildNode == None:
                UCB_Values.append(-1000000)  
        
        max = -1000000
        for x in range(len(UCB_Values)):
            temp = UCB_Values[x]
            if temp > max:
                max = temp
                index = x
        return index