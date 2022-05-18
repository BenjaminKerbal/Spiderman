import math
import random
import pymunk
from Environment.Utils.BodyUtils import BodyUtils
from Environment.Utils.MathUtils import MathUtils

from Environment.Utils.ObjectShapes import ObjectShapes
from Environment.Utils.ObjectType import ObjectType


class PlayerController:

    def __init__(self, env_space, start_point):
        self.player_size = 20
        self.env_space = env_space
        self.object = self.__create_player(self.player_size, start_point)
        self.max_webs = 2
        self.lines = []
        self.current_timestep = 0
        self.max_speed = 600
        self.max_web_range = 1000

    # -------------------------------------------------------------------------
    # Private functions

    def __create_player(self, size, start_point):
        x_variation = int(random.random() * 150) - 50
        y_variation = int(random.random() * 100) - 40
        self.body = pymunk.Body()
        body = pymunk.Body(1, 100, body_type= pymunk.Body.DYNAMIC)
        body.position = (start_point[0] + x_variation, start_point[1] + y_variation)
        shape = pymunk.Circle(body, size)
        shape.collision_type = 1
        self.env_space.add(body, shape)
        return {
            "shape" : shape,
            "size" : size,
            "drawShape" : ObjectShapes.Circle,
            "type" : ObjectType.Player,
        }

    def __create_line(self, target, is_body=False, applied_time_step=None):
        if not is_body:
            target_pos = target
            target = pymunk.Body(body_type=pymunk.Body.STATIC)
            target.position = target_pos
        distance = BodyUtils.distance_between_two_bodies_center(self.object['shape'].body, target)
        #joint = pymunk.DampedSpring(self.player['shape'].body, target, (0,0), (0,0), distance, 10, 1)
        joint = pymunk.SlideJoint(self.object['shape'].body, target, (0,0), (0,0), 0, distance)
        self.env_space.add(joint)
        applied_time_step = self.current_timestep if applied_time_step is None else applied_time_step
        return {
            "shape" : joint,
            "size" : 4,
            "drawShape" : ObjectShapes.Line,
            "type" : ObjectType.Net,
            "target_point" : target_pos,
            "applied_time_step" : applied_time_step
        }

    def __get_rays_from_body(self):
        body = self.object['shape'].body
        body_position = body.position
        number_of_rays = 24
        rays_list = []
        for i in range(number_of_rays):
            angle = math.radians(360/(number_of_rays) * i)
            new_x = body_position[0]+ int(math.cos(angle) * self.max_web_range)
            new_y = body_position[1]+ int(math.sin(angle) * self.max_web_range)
            segment_q = self.env_space.segment_query(body_position, (new_x, new_y), 1, pymunk.ShapeFilter())
            segment_q = [segment for segment in segment_q if segment.shape.body.position != body.position]
            if len(segment_q) == 0:
                rays_list.append(1)
            else:
                min_distance = 10000
                for segment in segment_q:
                    distance = round(BodyUtils.distance_between_body_and_point(body, segment.point), 3)
                    if distance < min_distance:
                        min_distance = distance
                rays_list.append(min_distance / self.max_web_range)
        return rays_list

    def __get_net_connection_points(self):
        body : pymunk.Body = self.get_body()
        connection_points = [-1, -1, -1, -1, -1, -1]
        if len(self.lines) == 1:
            angle = MathUtils.angle_between_two_positions(body.position, self.lines[0]['target_point'])
            connection_points[0] = (math.sin(angle) + 1) / 2
            connection_points[1] = (math.cos(angle) + 1) / 2
            connection_points[2] = BodyUtils.distance_between_body_and_point(body, self.lines[0]['target_point']) / self.max_web_range
        elif len(self.lines) >= 2:
            for i, net in enumerate(self.lines):
                angle = MathUtils.angle_between_two_positions(body.position, net['target_point'])
                connection_points[3*i] = (math.sin(angle) + 1) / 2
                connection_points[3*i+1] = (math.cos(angle) + 1) / 2
                connection_points[3*i+2] = BodyUtils.distance_between_body_and_point(body, net['target_point']) / self.max_web_range
        return connection_points
    
    def __net_release_pos(self):
        release_net_pos = None
        if len(self.lines) == 0:
            return None
        if len(self.lines) == 1:
            line = self.lines[0]
            release_net_pos = line['target_point']
            self.env_space.remove(line['shape'])
            self.lines.pop(0)
            return release_net_pos
        else:
            min_time_step = self.current_timestep + 1
            earliest_line = None
            earliest_i = None
            for i in range(len(self.lines)):
                line = self.lines[i]
                if line['applied_time_step'] < min_time_step:
                    min_time_step = line['applied_time_step']
                    earliest_line = line
                    earliest_i = i
            self.env_space.remove(earliest_line['shape'])
            self.lines.pop(earliest_i)
            return None

    def __set_pullup_speed(self, body, release_net_pos):
        velocity = body.velocity
        angle = math.degrees(MathUtils.angle_between_two_positions(body.position, release_net_pos))
        if angle >= 85 and angle <= 105:
            min_speed = 50
            max_speed = 150
            if velocity[1] > max_speed:
                return False
            new_y_speed = -max(min(abs(velocity[1])*1.1, min_speed), max_speed)
            body.velocity = (velocity[0], new_y_speed)
            return True
        else:
            return False

    # -------------------------------------------------------------------------
    # Getters

    def get_pos(self):
        return self.object['shape'].body.position
    
    def get_body(self):
        return self.object['shape'].body

    def get_observation(self):
        observation = list((self.__get_net_connection_points()))
        observation.extend(self.__get_rays_from_body())
        return [round(ob, 3) for ob in observation]

    # -------------------------------------------------------------------------
    # Net functions
    
    def can_eject_more_nets(self):
        return len(self.lines) < 2

    def release_net_and_apply_force(self):
        release_net_pos = self.__net_release_pos()
        if release_net_pos is None:
            return
        body : pymunk.Body = self.object['shape'].body
        velocity_vector = body.velocity
        if velocity_vector[0] < 20:
            if self.__set_pullup_speed(body, release_net_pos):
                return
        
        speed = math.sqrt(velocity_vector[0]**2 + velocity_vector[1]**2)
        angle = math.atan2(velocity_vector[1], velocity_vector[0])
        if speed > self.max_speed:
            speed = self.max_speed
        new_speed = min(speed * 1.25, self.max_speed)
        x_speed = int(math.cos(angle) * new_speed)
        y_speed = int(math.sin(angle) * new_speed)
        body.velocity = (x_speed, y_speed)


    def eject_net(self, target, point=False):
        body = self.object['shape'].body
        body_position = body.position
        if point:
            angle = math.atan2(target[1]-body_position[1], target[0]-body_position[0])
        else:
            angle = target
        new_x = body_position[0]+ int(math.cos(angle) * self.max_web_range)
        new_y = body_position[1]+ int(math.sin(angle) * self.max_web_range)
        #segment_q = self.env_space.segment_query_first(body_position, (new_x, new_y), 1, pymunk.ShapeFilter(categories=0b01))
        segment_q = self.env_space.segment_query(body_position, (new_x, new_y), 1, pymunk.ShapeFilter())
        segment_q = [segment for segment in segment_q if segment.shape.body.position != body.position]
        first_point = None
        min_distance = 10000
        for segment in segment_q:
            distance = BodyUtils.distance_between_body_and_point(body, segment.point)
            if distance < min_distance:
                min_distance = distance
                first_point = segment.point
        if first_point is not None:
            self.lines.append(self.__create_line(first_point))
        return first_point is None

    def step_nets(self):
        if len(self.lines) == 0:
            return
        body = self.object['shape'].body
        body_position = body.position
        for i, line in enumerate(self.lines):
            old_target_point = line['target_point']
            min_distance = BodyUtils.distance_between_body_and_point(body, old_target_point)
            best_segment = None
            segment_q = self.env_space.segment_query(body_position, old_target_point, 1, pymunk.ShapeFilter())
            segment_q = [segment for segment in segment_q if segment.shape.body.position != body.position]
            if len(segment_q) == 2 and len(set([segment.point for segment in segment_q])) == 1:
                segment_q = [segment_q[0]]
            for segment in segment_q:
                distance = BodyUtils.distance_between_body_and_point(body, segment.point)
                if distance < min_distance:
                    min_distance = distance
                    best_segment = segment
            if best_segment is not None:
                self.env_space.remove(line['shape'])
                self.lines.pop(i)
                self.lines.append(self.__create_line(best_segment.point, applied_time_step=line['applied_time_step']))
                if len(self.lines) > 1:
                    self.lines = sorted(self.lines, key=lambda k: k['applied_time_step'])       
                break
        self.current_timestep += 1

    