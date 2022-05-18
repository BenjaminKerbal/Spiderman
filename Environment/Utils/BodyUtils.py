import pymunk
import math


class BodyUtils:

    @staticmethod
    def distance_between_body_and_point(body1 : pymunk.Body, point):
        body1_pos = body1.position
        distance = math.sqrt((body1_pos[0] - point[0])**2 + (body1_pos[1] - point[1])**2)
        return distance

    @staticmethod
    def distance_between_two_bodies_center(body1 : pymunk.Body, body2 : pymunk.Body):
        body1_pos = body1.position
        body2_pos = body2.position
        distance = math.sqrt((body1_pos[0] - body2_pos[0])**2 + (body1_pos[1] - body2_pos[1])**2)
        return distance

    @staticmethod
    def distance_between_two_bodies_closest(body1 : pymunk.Body, body2 : pymunk.Body, body1_size=None, body2_size=None):
        compute_class = ComputeClosestDistance()
        return compute_class.compute_closest_distance(body1, body2, body1_size, body2_size)




# Helper class

class ComputeClosestDistance:

    def __get_body_points(self, body : pymunk.Body, size):
        center_posistion = body.position
        if size is None:
            corners = list(body.shapes)[0].bb
            return {
                "center" : center_posistion,
                "down_left" : (corners.left, corners.top),
                "down_right" : (corners.right, corners.top),
                "up_left" : (corners.left, corners.bottom),
                "up_right" : (corners.right, corners.bottom)
            }
        return {
            "center" : center_posistion,
            "down_left" : (center_posistion[0] - size[0]//2, center_posistion[1] + size[1]//2),
            "down_right" : (center_posistion[0] + size[0]//2, center_posistion[1] + size[1]//2),
            "up_left" : (center_posistion[0] - size[0]//2, center_posistion[1] - size[1]//2),
            "up_right" : (center_posistion[0] + size[0]//2, center_posistion[1] - size[1]//2)
        }

    def body1_relative_to_body2(self, body1_pos, body2_pos):
        x_pos = None
        y_pos = None
        if body1_pos['up_right'][0] < body2_pos['up_left'][0]:
            x_pos = "left"
        elif body1_pos['up_left'][0] > body2_pos['up_right'][0]:
            x_pos = "right"
        else:
            x_pos = "middle"
        
        if body1_pos['up_left'][1] > body2_pos['down_left'][1]:
            y_pos = "down"
        elif body1_pos['down_left'][1] < body2_pos['up_left'][1]:
            y_pos = "up"
        else:
            y_pos = "middle"
        return x_pos, y_pos

    def get_closest_distance(self, body1_points, body2_points, side):
        if side[0] == "middle":
            if side[1] == "up":
                return abs(body1_points['down_left'][1] - body2_points['up_left'][1])
            if side[1] == "down":
                return abs(body1_points['up_left'][1] - body2_points['down_left'][1])
            return 0
        if side[1] == "middle":
            if side[0] == "left":
                return abs(body1_points['up_right'][0] - body2_points['up_left'][0])
            if side[0] == "right":
                return abs(body1_points['up_left'][0] - body2_points['up_right'][0])
            return 0
        body2_corner_point = body2_points[side[1] + "_" + side[0]]
        body1_x_corner = "left" if side[0] == "right" else "right"
        body1_y_corner = "down" if side[1] == "up" else "up"
        body1_start_point = body1_points[body1_y_corner + "_" + body1_x_corner]
        return math.sqrt((body1_start_point[0] - body2_corner_point[0])**2 + (body1_start_point[1] - body2_corner_point[1])**2)



    def compute_closest_distance(self, body1 : pymunk.Body, body2 : pymunk.Body, body1_size=None, body2_size=None):
        body1_points = self.__get_body_points(body1, body1_size)
        body2_points = self.__get_body_points(body2, body2_size)
        side = self.body1_relative_to_body2(body1_points, body2_points)
        distance = self.get_closest_distance(body1_points, body2_points, side)
        return distance

        