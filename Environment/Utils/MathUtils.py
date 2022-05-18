
import math




class MathUtils:

    @staticmethod
    def angle_between_two_positions(pos1, pos2):
        return math.atan2(pos1[1]-pos2[1], pos1[0]-pos2[0])

