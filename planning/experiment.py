from mcts import Node
from episode import Episode
from timeit import default_timer as timer
from statistics import STATISTICS

from pathlib2 import Path
import pandas as pd


class Results:
    def __init__(self):
        self.Time = STATISTICS(0., 0.)
        self.Reward = STATISTICS(0., 0.)
        self.DiscountedReturn = STATISTICS(0., 0.)
        self.UndiscountedReturn = STATISTICS(0., 0.)
        self.Steps = STATISTICS(0., 0.)

    def Clear(self):
        self.Time.Clear()
        self.Reward.Clear()
        self.DiscountedReturn.Clear()
        self.UndiscountedReturn.Clear()
        self.Steps.Clear()



class ExperimentParams:
    SpawnArea = 4
    NumRuns = 10
    
    NumSteps = 100
    SimSteps = 1000
    TimeOut = 36000
    MinDoubles = 0
    MaxDoubles = 20
    NumDepth = 14
    TransformDoubles = -1
    TransformAttempts = 1000
    Accuracy = 0.01
    UndiscountedHorizon = 100
    AutoExploration = True
    
    NodeNum = 5000


class Experiment:
    def __init__(self, real, simulator):
        self.Real = real
        self.Simulator = simulator
        self.Episode = Episode()

    def Run(self, state):
        Root = Node(state, self.Simulator)
        Root.ExpandNode(state)
        #print(Root.Child(4).GameState.AgentPos.X)
        #print(Root.Child(4).GameState.AgentPos.Y)
        start = timer()
        t = 0
        
        while t < ExperimentParams.NodeNum:
            #print(t)
            node = Root.selection(state)
            #print(node.GameState.AgentPos.X)
            #print(node.GameState.AgentPos.Y)
            totalReward = node.Simulation(node.GameState)
            node.BackPropagation(totalReward)
            node = node.ExpandNode(node.GameState)
            t += 1
            
        bestAction = Root.SelectAction(state)
        bestChild = Root.Child(bestAction)
        return bestAction
        
    def DiscountedReturn(self, simulationDirectory, knowledge):
        directory = simulationDirectory + '/Depth_%d' % (ExperimentParams.NodeNum)
        Path(directory).mkdir(parents=True, exist_ok=True)
        print('/Depth_%d' % (ExperimentParams.NodeNum) + "created")
        
        for trial in range(ExperimentParams.MinDoubles, ExperimentParams.MaxDoubles):
            episodeFile = directory + '/Episode_%d.csv' % (trial)

            #self.Results.Clear()
            state = self.Real.CreateStartState()
            copyState = self.Real.Copy(state)
            self.Episode.Add(-1, copyState, 0)
            #self.Episode.Complete()
            #print(state.AgentPos.X)
            #print(state.AgentPos.Y)
            t = 0
            while t < ExperimentParams.NumSteps:
                currentState = self.Real.Copy(state)
                bestAction = self.Run(state)
                terminal, state, reward = self.Real.Step(currentState, bestAction)
                if terminal:
                    self.Episode.Add(bestAction, state, reward)
                    self.Episode.Complete()
                    break

                self.Episode.Add(bestAction, state, reward)
                t += 1
                
            self.Episode.Episode2CSV(episodeFile)
            self.Episode.Clear()
            
            