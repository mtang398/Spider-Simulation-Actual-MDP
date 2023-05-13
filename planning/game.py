from simulator import Simulator
from grid import Grid
from coord import COORD, COMPASS, Compass, CompassString, Opposite, AggressiveDirectionalDistance, ManhattanDistance
from utils import Random, Bernoulli, SetFlag

from math import floor
import numpy as np
import random

class GameState:
    def __init__(self):
        self.AgentPos = COORD(7, 0)
        self.PreyPos = COORD(7, 7)
        self.PreyDir = -1
        self.PreySpeedMult = 1
        self.AgentObservationDirection = 0
        self.PreyPreviousPos = COORD(7,7)
        self.AgentPreviousPos = COORD(7,0)
        self.Depth = 0

    def __eq__(self, other):
        return self.AgentPos == other.AgentPos and self.PreyPos == other.PreyPos

    def __str__(self):
        return "(" + str(self.AgentPos) + "; " + str(self.PreyPos) + ")"

    def __hash__(self):
        return 0
    
    def copy(self):
        newState = GameState()

        newState.AgentPos = self.AgentPos
        newState.PreyPos = self.PreyPos
        newState.PreyDir = self.PreyDir
        newState.PreySpeedMult = self.PreySpeedMult
        newState.Depth = self.Depth
        newState.AgentObservationDirection = self.AgentObservationDirection
        newState.PreyPreviousPos = self.PreyPreviousPos
        newState.AgentPreviousPos = self.AgentPreviousPos
        
        return newState

class Game(Simulator):
    def __init__(self, xsize, ysize, verbose=False):
        listStarts = [COORD(0,0),COORD(0,14),COORD(14,0),COORD(14,14)]
        self.AgentHome = COORD(7,0)
        self.PreyNum = 1
        self.PreyHome = COORD(7, 7)
        self.TurnLeftProbability = 0.1
        self.TurnRightProbability = 0.1
        self.MoveProbability = 0.2
        self.Discount = 0.95

        self.Grid = Grid(xsize, ysize)

        self.NumActions = 5

        Simulator.__init__(self, self.NumActions, self.Discount)

        self.RewardDefault = -1
        self.RewardStay = 0.2
        self.RewardCatchPreyFront = -149
        self.RewardCatchPreySide = 51
        self.RewardCatchPreyBack = 101
        self.RewardCatchPreyWhileStatic = 50
        self.RewardRange = 1000
        self.RewardRight = 0.5
        self.RewardHitWall = -500
        self.RewardSpiral = 0.5
        self.RewardSpiral2 = 0.3
        self.PenaltySameBlcok = -2

        self.State = GameState()
        self.State.AgentPos = self.AgentHome
        self.State.PreyPos = self.PreyHome
        self.State.PreyPreviousPos = self.PreyHome
        self.AgentPreviousPos = COORD(14,14)

        invalidPreyLocations = [self.AgentHome]

        allPreyLocations = [COORD(x, y) for x in range(0, xsize) for y in range(0, ysize)]
        validPreyLocations = list(set(allPreyLocations) - set(invalidPreyLocations))
        self.StartPreyLocations = validPreyLocations

        self.XSize = xsize
        self.YSize = ysize
        if verbose:
            self.InitializeDisplay()

    def FreeState(self, state):
        del state

    def InitializeDisplay(self):
        state = self.CreateStartState()
        self.DisplayState(state)

    def Copy(self, state):
        newState = GameState()

        newState.AgentPos = state.AgentPos
        newState.PreyPos = state.PreyPos
        newState.PreyDir = state.PreyDir
        newState.PreySpeedMult = state.PreySpeedMult
        newState.Depth = state.Depth
        newState.AgentObservationDirection = state.AgentObservationDirection
        newState.PreyPreviousPos = state.PreyPreviousPos
        newState.AgentPreviousPos = state.AgentPreviousPos

        return newState

    def Validate(self, state):
        assert(self.Grid.Inside(state.AgentPos))
        assert(self.Grid.Inside(state.PreyPos))

    def CreateStartState(self):
        state = GameState()
        state = self.NewLevel(state)
        return state

    def CreateRandomStartState(self):
        state = GameState()
        state = self.NewLevel(state)

        state.PreyPos = self.StartPreyLocations[Random(0, len(self.StartPreyLocations))]
        state.PreyPreviousPos = self.StartPreyLocations[Random(0, len(self.StartPreyLocations))]
        return state

    def NextPos(self, fromCoord, dir):
        nextPos = fromCoord + Compass[dir]
        if self.Grid.Inside(nextPos):
            return nextPos
        else:
            return Compass[COMPASS.NAA]

    def Step(self, state, action):
        oldpos = state.AgentPreviousPos
        currentpos = state.AgentPos
        state.AgentPreviousPos = currentpos
        if action < 4:
            state.AgentObservationDirection = action
        reward = self.RewardDefault
        
        newpos = self.NextPos(state.AgentPos, action)
        
        if newpos.Valid():
            state.AgentPos = newpos
        else:
            reward += self.RewardHitWall
            
        if action == 4:
            reward += self.RewardStay

        #print('new pos')
        #print(state.AgentPos)
        #print('current pos')
        #print(state.AgentPreviousPos)
        #print('old pos')
        #print(oldpos)
        copyState = self.Copy(state)
        previousPreyLocation = copyState.PreyPreviousPos
        state = self.CockroachTurn(state)
        direction = 0
        if action < 4:
            direction = action
            
        for x in range (state.PreyPos.X -3, state.PreyPos.X + 3):
            for y in range (state.PreyPos.Y -3, state.PreyPos.Y + 3):
                if state.AgentPos == COORD(x,y):
                    reward += self.RewardSpiral
                    if action == 0 and state.AgentPos.X > state.PreyPos.X and state.AgentPos.Y < state.PreyPos.Y:
                        reward += self.RewardRight
                    elif action == 1 and state.AgentPos.X < state.PreyPos.X and state.AgentPos.Y < state.PreyPos.Y:
                        reward += self.RewardRight
                    elif action == 2 and state.AgentPos.X > state.PreyPos.X and state.AgentPos.Y > state.PreyPos.Y:
                        reward += self.RewardRight
                    elif action == 3 and state.AgentPos.X < state.PreyPos.X and state.AgentPos.Y > state.PreyPos.Y:
                        reward += self.RewardRight
                    if state.AgentPos == oldpos:
                        reward += self.PenaltySameBlcok
        for x in range (state.PreyPos.X -1, state.PreyPos.X + 1):
            for y in range (state.PreyPos.Y -1, state.PreyPos.Y + 1):
                if state.AgentPos == COORD(x,y):
                    reward += self.RewardSpiral2
            
        if state.AgentPos == state.PreyPos:
            if state.PreyDir == action:
                reward += self.RewardCatchPreyBack
                if previousPreyLocation == state.PreyPos and state.PreyPos == state.PreyPreviousPos:
                    reward += self.RewardCatchPreyWhileStatic
                return True, state,  reward 
            elif abs(state.PreyDir - action) == 1 or abs(state.PreyDir - action) == 3:
                reward += self.RewardCatchPreySide
                if previousPreyLocation == state.PreyPos and state.PreyPos == state.PreyPreviousPos:
                    reward += self.RewardCatchPreyWhileStatic
                return True, state,  reward
            elif abs(state.PreyDir - action) == 2:
                reward += self.RewardCatchPreyFront
                if previousPreyLocation == state.PreyPos and state.PreyPos == state.PreyPreviousPos:
                    reward += self.RewardCatchPreyWhileStatic
                return True, state, reward
            
        state.Depth += 1
        return False, state, reward
    
    def NewLevel(self, state):
        state.AgentPos = self.AgentHome
        state.PreyPos = self.PreyHome
        state.PreyPreviousPos = self.PreyHome
        state.AgentPreviousPos = COORD(7,0)
        state.PreyDir = 2
        state.Depth = 0
        state.AgentObservationDirection = 0

        return state


    def GenerateLegal(self, state, history, legal, status):
        for a in range(self.NumActions):
            newpos = self.NextPos(state.AgentPos, a)
            if newpos.Valid():
                legal.append(a)

        return legal

    def GeneratePreferred(self, state, history, actions, status):
        if history.Size():
            for a in range(self.NumActions):
                newpos = self.NextPos(state.AgentPos, a)
                if newpos.Valid():
                    actions.append(a)

            if Opposite(history.Back().Action) in actions:
                actions.remove(Opposite(history.Back().Action))
            #actions = self.GenerateExplorationActions(state, history, actions, status)

            return actions

        else:
            return self.GenerateLegal(state, history, actions, status)

    def DisplayAction(self, state, action):
        print("Agent moves ", CompassString[action])
        
    def moveCockroach(self, state):
        rand = random.random()
        if rand <= 1:
            copyState = self.Copy(state)
            copyState.PreyPreviousPos = state.PreyPos
            return copyState
        if rand <= 3 and rand > 1:
            return self.LeftTurn(state)
        if rand <= 5 and rand > 3:
            return self.RightTurn(state)
        if rand > 5:
            return self.Move(state)
    
    def LeftTurn(self, state):
        copyState = self.Copy(state)
        copyState.PreyPreviousPos = state.PreyPos
        if copyState.PreyDir != 0:
            copyState.PreyDir -= 1
        elif copyState.PreyDir == 0:
            copyState.PreyDir = 3
        return copyState
    
    def RightTurn(self, state):
        copyState = self.Copy(state) 
        copyState.PreyPreviousPos = state.PreyPos
        if copyState.PreyDir != 3:
            copyState.PreyDir += 1
        elif copyState.PreyDir == 3:
            copyState.PreyDir = 0
        return copyState
    
    def Move(self, state):
        newpos = self.NextPos(state.PreyPos, state.PreyDir)
        if newpos.Valid() and newpos != state.AgentPos:
            copyState = self.Copy(state)
            copyState.PreyPreviousPos = state.PreyPos
            copyState.PreyPos = newpos
            return copyState
        else:
            return self.moveCockroach(state)
        
    def CockroachTurn(self,state):
        copyState = self.Copy(state) 
        copyState.PreyPreviousPos = state.PreyPos
        rand = random.random()
        if rand <= 0.5:
            return copyState
        else:
            if copyState.AgentPos.X > copyState.AgentPos.Y and (copyState.AgentPos.X + copyState.AgentPos.Y) < 14:
                if copyState.PreyDir == 2:
                    return copyState
                elif copyState.PreyDir == 0 or copyState.PreyDir == 1:
                    copyState.PreyDir = copyState.PreyDir + 1
                elif copyState.PreyDir == 3:
                    copyState.PreyDir = copyState.PreyDir - 1
            elif copyState.AgentPos.X > copyState.AgentPos.Y and (copyState.AgentPos.X + copyState.AgentPos.Y) > 14:
                if copyState.PreyDir == 1:
                    return copyState
                elif copyState.PreyDir == 3 or copyState.PreyDir == 2:
                    copyState.PreyDir = copyState.PreyDir - 1
                elif copyState.PreyDir == 0:
                    copyState.PreyDir = copyState.PreyDir + 1
            elif copyState.AgentPos.X < copyState.AgentPos.Y and (copyState.AgentPos.X + copyState.AgentPos.Y) < 14:
                if copyState.PreyDir == 3:
                    return copyState
                elif copyState.PreyDir == 1 or copyState.PreyDir == 2:
                    copyState.PreyDir = copyState.PreyDir + 1
                elif copyState.PreyDir == 0:
                    copyState.PreyDir = copyState.PreyDir + 3
            elif copyState.AgentPos.X < copyState.AgentPos.Y and (copyState.AgentPos.X + copyState.AgentPos.Y) > 14:
                if copyState.PreyDir == 0:
                    return copyState
                elif copyState.PreyDir == 1 or copyState.PreyDir == 2:
                    copyState.PreyDir = copyState.PreyDir - 1
                elif copyState.PreyDir == 3:
                    copyState.PreyDir = copyState.PreyDir - 3
            return copyState

if __name__ == "__main__":
    state = GameState()
    game = Game(7, 7)
    game.Validate(state)
