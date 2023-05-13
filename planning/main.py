from msilib.schema import Directory
from game import Game
from simulator import Knowledge
from experiment import ExperimentParams, Experiment
from utils import UnitTestUTILS, Random
from coord import COORD, UnitTestCOORD
from grid import Grid
from math import floor
import os, sys, pickle, itertools
from pathlib2 import Path
import numpy as np
import pandas as pd

sys.setrecursionlimit(10000)

XSize = 15
YSize = 15

treeknowlege = 2 # 0 = pure, 1 = legal, 2 = smart
rolloutknowledge = 2 # 0 = pure, 1 = legal, 2 = smart
smarttreecount = 10.0 # prior count for preferred actions in smart tree search
smarttreevalue = 1.0 # prior value for preferred actions during smart tree search

real = Game(XSize, YSize)
simulator = Game(XSize, YSize)

knowledge = Knowledge()
knowledge.TreeLevel = treeknowlege
knowledge.RolloutLevel = rolloutknowledge
knowledge.SmartTreeCount = smarttreecount
knowledge.SmartTreeValue = smarttreevalue

experiment = Experiment(real, simulator)


directory = os.getcwd()
simulationDirectory = directory + '/Data'
Path(simulationDirectory).mkdir(parents=True, exist_ok=True)
print("file created")

action, state, reward = experiment.DiscountedReturn(simulationDirectory, knowledge)