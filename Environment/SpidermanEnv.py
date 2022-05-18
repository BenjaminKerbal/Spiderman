import pymunk
import math
import numpy as np
from collections import deque
import copy
from Environment.Camera import Auto, Camera
from Environment.ObstacleGenerator import ObstacleGenerator
from Environment.PlayerController import PlayerController
from Environment.Rendering.GraphicsRenderer import GraphicsRenderer
from Environment.Rendering.LegacyRenderer import LegacyRenderer
from Environment.Utils.BodyUtils import BodyUtils


class SpidermanEnv:

    SCORE_BAR_SIZE = 85
    USE_LEGACY_RENDERER = False

    def __init__(self, eval=False, render=False, max_steps=20000):
        display_size = (1300, 800)
        self.game_size = (display_size[0], display_size[1] - self.SCORE_BAR_SIZE)
        self.environment_update_intervall = 1/50
        self.eval = eval
        self.skipp_frames = 5
        self.max_steps = max_steps
        self.debug_pos = False

        self.high_score = 0
        self.max_speed_reward = 0.5
        self.web_shooter_max_ammo = 10
        self.web_shooter_reload_speed_multiplier = 1.2 # Default 0.8
        self.web_shooter_current_ammo = self.web_shooter_max_ammo
        self.reset()
        self.observation_space = self.get_observation_size()
        self.rope_actions = 12
        self.action_space = self.rope_actions + 2
        self.render_screen = render
        self.renderer = None
        if render:
            if self.USE_LEGACY_RENDERER:
                self.renderer = LegacyRenderer(display_size, self.SCORE_BAR_SIZE)
            else:
                self.renderer = GraphicsRenderer(display_size, self.SCORE_BAR_SIZE)


    def reset(self):
        self.env_space = pymunk.Space()
        self.env_space.gravity = (0, 150)
        self.obstaclegenerator = ObstacleGenerator(self.env_space, self.game_size)
        self.player = PlayerController(self.env_space, (150, int((self.game_size[1]//5))))
        self.roof = []
        self.obstacles = []
        self.score = 0
        self.current_timestep = 0
        self.scroll_timestep = 0
        self.start_scroll = False
        self.first_net_shot = False
        self.camera = Camera(self.player, self.game_size)
        # self.follow = Follow(self.camera, self.player)
        self.auto = Auto(self.camera, self.player)
        self.camera.set_method(self.auto)
        self.web_shooter_current_ammo = self.web_shooter_max_ammo
        self.limit_actions = False

        # Rewards
        self.touching_an_object = False
        self.last_score_given_reward = 0
        # Debug
        self.debug_obstacles = None
        if self.debug_pos:
            self.position_history = 30
            self.last_position_list = deque([])

        return self.get_observation()

    # Debug
    def get_closes_obstacle(self, point):
        min_distance = 1000000
        closest_obstacle = None
        for obstacle in self.obstacles:
            distance = BodyUtils.distance_between_body_and_point(obstacle['shape'].body, point)
            if distance < min_distance:
                closest_obstacle = obstacle
                min_distance = distance
        return closest_obstacle

    def get_observation_size(self):
        return self.get_observation().shape[1]

    def update_score(self):
        current_score = int(self.player.get_pos().x / 100)
        if current_score > self.score:
            self.score = current_score
            if self.score > self.high_score:
                self.high_score = self.score

    def get_observation(self):
        body : pymunk.Body = self.player.get_body()
        observation = [
            self.web_shooter_current_ammo / self.web_shooter_max_ammo,
            body.velocity.x / (self.player.max_speed * 1.3),
            body.velocity.y / (self.player.max_speed * 1.3),
            math.sqrt(body.velocity.x**2 + body.velocity.y**2) / (self.player.max_speed * 1.3),
            (body.position.x - self.camera.offset.x) / self.camera.CONST.x,
            body.position.y / self.game_size[1]
        ]
        observation.extend(self.player.get_observation())
        return np.array(observation).reshape(1, -1)


    ''' Colliders not currently used '''
    def collide(self, arbiter, space, data):
        self.touching_an_object = True
        return True

    def release(self, arbiter, space, data):
        self.touching_an_object = False
        
    def check_if_player_touches_ground(self):
        collision_handler = self.env_space.add_collision_handler(1, 2)
        collision_handler.begin = self.collide
        collision_handler.separate = self.release


    def check_lose(self):
        position = self.player.get_pos()
        return position.y > self.game_size[1] or position.x + self.player.player_size*0.2 < self.camera.offset.x
    
    # Speed and survive
    def get_reward_and_done(self):
        if self.check_lose():
            return -350, True
        body = self.player.get_body()
        reward = 0

        if self.launched_net:
            reward -= 0.5
            self.launched_net = False

        velocity = body.velocity.x
        if velocity > 290:
            reward += 1 / self.skipp_frames
        if velocity > 230:
            reward += 0.7 / self.skipp_frames
        if velocity > 180:
            reward += 0.3 / self.skipp_frames
        elif velocity > 90:
            reward += 0.1 / self.skipp_frames
        elif abs(velocity) < 10:
            reward -= 0.1 / self.skipp_frames
        return reward, False
    
    def __step_forward(self, action):
        if self.first_net_shot == False:
            self.first_net_shot = not self.player.eject_net(math.radians(270), point=False)
        if action != 0:
            self.start_scroll = True
            if action == 1:
                if len(self.player.lines) != 0:
                    self.player.release_net_and_apply_force()
            else:
                if len(self.player.lines) < 2  and self.web_shooter_current_ammo >= 1:
                    self.launched_net = True
                    self.web_shooter_current_ammo -= 1
                    angle = (360 // self.rope_actions) * (action - 2)
                    self.missed_net = self.player.eject_net(math.radians(angle))
        
        if self.start_scroll:
            self.camera.scroll()
            self.scroll_timestep += 1
            if (self.scroll_timestep + 1) % 450 == 0 and isinstance(self.camera.method, Auto):
                self.camera.method.set_scroll_speed(self.camera.method.last_scroll_speed + 0.1)
        self.roof, self.obstacles = self.obstaclegenerator.step(self.camera.offset.x)
        self.player.step_nets()
        self.update_score()
        self.web_shooter_current_ammo += self.web_shooter_reload_speed_multiplier * self.environment_update_intervall
        self.web_shooter_current_ammo = round(min(self.web_shooter_current_ammo, self.web_shooter_max_ammo), 3)
        self.env_space.step(self.environment_update_intervall)
        return self.get_reward_and_done()

    def step(self, actions):
        self.launched_net = False
        self.missed_net = False
        total_reward = 0

        # Skipp frames to speed up learning
        for _ in range(self.skipp_frames):
            if self.render_screen:
                self.render()
            reward, done = self.__step_forward(actions)
            actions = 0
            total_reward += reward
            if done:
                break
        
        self.current_timestep += 1
        observation = self.get_observation()
        self.limit_actions = len(self.player.lines) >= 2
        info = {
         'score' : self.score
        }
        
        if self.debug_pos:
            if len(self.last_position_list) < self.position_history:
                self.last_position_list.append(copy.deepcopy(self.player.object))
            else:
                self.last_position_list.append(copy.deepcopy(self.player.object))
                self.last_position_list.popleft()

        if self.current_timestep > self.max_steps:
            done = True

        # if done:
        #     print("timesteps:", self.current_timestep)
        return observation, total_reward, done, info

    def render(self):
        if self.renderer is None:
            raise Exception("environment must be set to visual in order to render")
        render_objects = [  
            self.obstacles,
            self.roof,
            self.player.lines,
            self.player.object,
        ]
        render_dict = {
            "camera" : self.camera,
            "render_objects" : render_objects,
            "web_shooter_ammo_percentage" : round(self.web_shooter_current_ammo / self.web_shooter_max_ammo, 3),
            "debug_string" : None 
        }
        debug = None
        if self.debug_pos and len(self.last_position_list) != 0:
            debug = self.last_position_list[0]
        self.renderer.render(render_dict, self.score, self.high_score, debug)

    
    