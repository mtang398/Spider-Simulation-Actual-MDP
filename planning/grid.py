from coord import COORD, COMPASS, ManhattanDistance, Compass
import numpy as np

class Grid:
    def __init__(self, xsize, ysize):
        self.XSize = xsize
        self.YSize = ysize
        self.Grid = [0]*(xsize * ysize)

    def GetXSize(self):
        return self.XSize

    def GetYSize(self):
        return self.YSize

    def Resize(self, xsize, ysize):
        self.Grid.append([0]*abs((self.XSize * self.YSize) - (xsize * ysize)))
        self.XSize = xsize
        self.YSize = ysize

    def Inside(self, coord):
        return coord.X >= 0 and coord.Y >= 0 and coord.X < self.XSize and coord.Y < self.YSize

    def Index(self, x, y):
        assert (self.Inside(COORD(x, y)))
        return ((self.XSize) * (self.YSize - 1 - y)) + x

    def Coordinate(self, index):
        assert (index < self.XSize * self.YSize)
        return COORD(divmod(index, self.XSize)[0], divmod(index, self.XSize)[1])
    
    def DistToEdge(self, coord, direction):
        assert(self.Inside(coord))
        return {
            1: self.YSize - 1 - coord.Y,
            2: self.XSize - 1 - coord.X,
            3: coord.Y,
            4: coord.X
        }.get(direction, False)

    def __getitem__(self, item):
        assert(item >= 0 and item < self.XSize * self.YSize)
        return self.Grid[item]