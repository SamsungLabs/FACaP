from torch import nn

class WallSegmentTerm(nn.Module):
    def __init__(self, wall, unproject, distance_function, floorplan):