from Environment.Utils.ObjectShapes import ObjectShapes
from Environment.Utils.ObjectType import ObjectType
from Environment.Utils.BodyUtils import BodyUtils
import pymunk
import random
import math


class ObstacleGenerator:

    ROOF_SIZE = 40

    def __init__(self, env_space, game_size):
        self.env_space = env_space
        self.game_size = game_size
        self.reduce_fist_obstacle = 1000
        self.min_size = (50, 50)
        self.max_size = (400, 400)
        self.min_empty_distance = 150
        self.max_obstacle_side_length = 600
        self.min_obstacle_side_length = 35
        self.obstacle_fequency_pixels = 200
        self.last_obstacle_addition = 0
        self.obstacle_collision_type = 2

        self.difficulty = 1
        self.roof_list = []
        self.obstacles_list = []
        # Start Wall
        # self.obstacles_list.append(self.__create_obstacle(ObjectShapes.Rectangle, (-100, self.game_size[1]//2), (150, self.game_size[1])))

        # Fist obstacles
        self.obstacles_list.append(self.__create_obstacle(ObjectShapes.Rectangle, (400, 300), (50, 200)))
        self.obstacles_list.append(self.__create_obstacle(ObjectShapes.Rectangle, (75, self.game_size[1] - 50), (50, 100)))
        self.obstacles_list.append(self.__create_obstacle(ObjectShapes.Rectangle, (1150, self.game_size[1] - 60), (50, 250)))



    def __delete_shape(self, object_to_remove):
        if isinstance(object_to_remove, dict):
            self.env_space.remove(object_to_remove['shape'], object_to_remove['shape'].body)
        else:
            self.env_space.remove(object_to_remove, object_to_remove.body)
    
    def __create_obstacle(self, draw_shape : ObjectShapes, pos : tuple, size):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = pos
        if draw_shape == ObjectShapes.Rectangle:
            shape = pymunk.Poly.create_box(body, size)
        elif draw_shape == ObjectShapes.Circle:
            shape = pymunk.Circle(body, size)
        else:
            raise("unrecognized shape")
        shape.collision_type = self.obstacle_collision_type
        self.env_space.add(body, shape)
        return {
            "shape" : shape,
            "size" : size,
            "drawShape" : draw_shape,
            "type" : ObjectType.Obstacle,
        }

    def __create_roof(self, left_bar):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (left_bar, self.ROOF_SIZE//2)
        shape = pymunk.Poly.create_box(body, (self.game_size[0]*2, self.ROOF_SIZE))
        shape.collision_type = self.obstacle_collision_type
        self.env_space.add(body, shape)
        return {
            "shape" : shape,
            "size" : (self.game_size[0]*2, self.ROOF_SIZE),
            "drawShape" : ObjectShapes.Rectangle,
            "type" : ObjectType.Roof,
        }

    def update_roof(self, left_bar):
        if len(self.roof_list) == 0:
            self.roof_list.append(self.__create_roof(left_bar))
            self.roof_list.append(self.__create_roof(left_bar + self.game_size[0]))
        if left_bar - self.roof_list[0]['shape'].body.position.x > self.game_size[0]:
            roof = self.roof_list.pop(0)
            self.__delete_shape(roof)
            self.roof_list.append(self.__create_roof(left_bar + self.game_size[0]))

    def create_random_object(self, left_bar):
        x_spawn_pos = left_bar + self.game_size[0] * 1.5
        if x_spawn_pos > self.obstacle_fequency_pixels + self.last_obstacle_addition:
            close_obstacle = [obstacle for obstacle in self.obstacles_list if obstacle['shape'].body.position.x > left_bar]
            shape = None
            attempts = 15
            for _ in range(attempts):
                x_length = int(random.random() * (self.max_obstacle_side_length - self.min_obstacle_side_length) + self.min_obstacle_side_length)
                y_length = int(max(random.random() * (self.max_obstacle_side_length - x_length - self.min_obstacle_side_length), 0) + self.min_obstacle_side_length)
                body = pymunk.Body(body_type=pymunk.Body.STATIC)
                random_y_pos = int(random.random() * self.game_size[1])
                body.position = (x_spawn_pos, random_y_pos)
                shape = pymunk.Poly.create_box(body, (x_length, y_length))
                no_close_obstacles = True
                for other_obstacle in close_obstacle:
                    distance = BodyUtils.distance_between_two_bodies_closest(shape.body, other_obstacle['shape'].body, body1_size=(x_length, y_length))
                    if distance < self.min_empty_distance:
                        no_close_obstacles = False
                        shape = None
                        break
                if no_close_obstacles:
                    shape.collision_type = self.obstacle_collision_type
                    self.env_space.add(body, shape)
                    self.obstacles_list.append({
                        "shape" : shape,
                        "size" : (x_length, y_length),
                        "drawShape" : ObjectShapes.Rectangle,
                        "type" : ObjectType.Obstacle,
                    })
                    self.last_obstacle_addition = x_spawn_pos
                    break

        
    def remove_old_obstacles(self, left_bar):
        for obstacles in self.obstacles_list:
            pos = obstacles['shape'].body.position
            if pos.x < left_bar - 300:
                if pos.x == -100 and pos.y == self.game_size[1]//2:
                    if pos.x < self.reduce_fist_obstacle + 100:
                        continue
                self.obstacles_list.remove(obstacles)
                self.__delete_shape(obstacles)

    def step(self, left_bar):
        self.update_roof(left_bar)
        self.remove_old_obstacles(left_bar)
        self.create_random_object(left_bar)
        return self.roof_list, self.obstacles_list
        