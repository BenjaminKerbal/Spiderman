import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
vec = pygame.math.Vector2
from abc import ABC, abstractmethod
import numpy as np

class Camera:

    def __init__(self, player, display_size):
        self.player = player
        self.offset = vec(0, 0)
        self.DISPLAY_W = display_size[0]
        self.DISPLAY_H = display_size[1]
        self.offset_float = vec(0, 0)
        self.CONST = vec(300,0)

    def set_method(self, method):
        self.method = method

    def scroll(self):
        self.method.scroll()

class CamScroll(ABC):

    def __init__(self, camera, player):
        self.camera = camera
        self.player = player

    @abstractmethod
    def scroll(self):
        pass

    @abstractmethod
    def set_scroll_speed(self, speed):
        pass

class Follow(CamScroll):

    def __init__(self, camera, player):
        CamScroll.__init__(self, camera, player)
    
    def set_scroll_speed(self, speed):
        pass        

    def scroll(self):
        player_pos = self.player.get_pos()
        self.camera.offset_float.x += (player_pos.x - self.camera.offset_float.x - self.camera.CONST.x)
        #self.camera.offset_float.y += (player_pos.y - self.camera.offset_float.y - self.camera.CONST.y)
        self.camera.offset.x, self.camera.offset.y = int(self.camera.offset_float.x), int(self.camera.offset_float.y)
        
class Auto(CamScroll):
    
    def __init__(self, camera, player):
        CamScroll.__init__(self, camera, player)
        self.max_speed = 2.4
        self.min_offset = None
        self.current_tick = 0
        self.float_speed = 0
        self.last_scroll_speed = -1
        self.set_scroll_speed(0.4)

    def set_scroll_speed(self, speed):
        if self.last_scroll_speed == speed:
            return
        if speed < 0.0:
            speed = 0.0
        elif speed > 1:
            speed = 1
        self.last_scroll_speed = round(speed, 2)
        self.float_speed = self.max_speed * speed

    def scroll(self):
        if self.min_offset is None:
            self.min_offset = self.camera.offset.x
        player_pos = self.player.get_pos()
        #self.camera.offset_float.x += (player_pos.x - self.camera.offset_float.x - self.camera.CONST.x)
        self.camera.offset_float.x = max(player_pos.x - self.camera.CONST.x, self.camera.offset_float.x + self.float_speed)
        self.camera.offset.x = round(self.camera.offset_float.x)

        