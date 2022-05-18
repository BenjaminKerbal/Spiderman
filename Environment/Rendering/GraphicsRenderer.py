
import math
import sys
import os
from os.path import join

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from Environment.Rendering.RenderingParent import RenderingParent
from Environment.Utils.ObjectShapes import ObjectShapes
from Environment.Utils.ObjectType import ObjectType
from Environment.Camera import Camera
from Environment.RenderSectionHandler import RenderSectionHandler

class GraphicsRenderer(RenderingParent):

    GRAPHICS_FOLDER = join(os.getcwd(), "Environment", "Rendering", "Graphics")

    def __init__(self, display_size, score_bar_size):
        pygame.init()
        
        
        self.screen = pygame.display.set_mode(display_size)
        self.clock = pygame.time.Clock()
        self.score_bar_size = score_bar_size

        self.background = self.__load_image("background.png")
        self.background_tuple = (0, score_bar_size, display_size[0], display_size[1])
        self.player_image = self.__load_image("player.png")
        self.obstacle_image = self.__load_image("obstacle.png")
        self.roof_image = (0, 0, 0) # No image
        self.net_image = (255, 255, 255) # No image
        
        self.score_background_color = (25, 25, 25)
        self.score_color = (255, 255, 255) # (200, 120, 230)
        self.score_font = pygame.font.SysFont('Comic Sans MS', 30)

    def __load_image(self, image_name):
        image = pygame.image.load(join(self.GRAPHICS_FOLDER, image_name)).convert_alpha()
        return image

    def __check_for_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def __get_pos_of_body(self, body, camera : Camera):
        positions = body.position
        return (int(positions[0] - camera.offset.x), int(positions[1]) + self.score_bar_size)

    def __get_angle_of_player(self, body):
        velocity = body.velocity
        return math.degrees(math.atan2(-velocity[1], velocity[0]))

    def __render_graphic_in_correct_place(self, image, pos, size, angle=-1):
        if not isinstance(size, tuple):
            size = (size, size)
        if angle == -1:
            self.screen.blit(image, pos, (0, 0, size[0], size[1]))
        else:
            top_left_postion = (pos[0] - size[0], pos[1] - size[1]) 
            rotated_image = pygame.transform.rotate(image, angle)
            new_rect = rotated_image.get_rect(center = image.get_rect(topleft = top_left_postion).center)   
            self.screen.blit(rotated_image, new_rect) 

    def __render_object(self, object, camera : Camera, override_color=None):
        if object['type'] == ObjectType.Player:
            image = self.player_image
        elif object['type'] == ObjectType.Roof:
            image = self.roof_image
        elif object['type'] == ObjectType.Obstacle:
            image = self.obstacle_image
        elif object['type'] == ObjectType.Net:
            image = self.net_image
        else:
            raise("Unrecognized type")
        
        if object['drawShape'] == ObjectShapes.Line:
            pos1 = self.__get_pos_of_body(object['shape'].a, camera)
            pos2 = self.__get_pos_of_body(object['shape'].b, camera)
            pygame.draw.line(self.screen, image, pos1, pos2, object['size'])
        else:
            pos = self.__get_pos_of_body(object['shape'].body, camera)
            if object['drawShape'] == ObjectShapes.Circle:
                angle = self.__get_angle_of_player(object['shape'].body)
                self.__render_graphic_in_correct_place(image, pos, object['size'], angle)
            elif object['drawShape'] == ObjectShapes.Rectangle:
                width = object['size'][0]
                height = object['size'][1]
                pos_x_start = pos[0] - width//2
                pos_y_start = pos[1] - height//2
                if not isinstance(image, tuple):
                    self.__render_graphic_in_correct_place(image, (pos_x_start, pos_y_start), object['size'])
                else:
                    pygame.draw.rect(self.screen, image, [pos_x_start, pos_y_start, width, height])
            else:
                raise("Unrecognized shape")

    def __render_objects(self, objects : dict):
        camera = objects['camera']
        for bodyList in objects['render_objects']:
            if isinstance(bodyList, list):
                for body in bodyList:
                    self.__render_object(body, camera)
            else:
                self.__render_object(bodyList, camera)


    def __render_text(self, text, value, pos):
        text = self.score_font.render(text + str(value), False, self.score_color)
        self.screen.blit(text, pos)

    def __render_web_shooter_ammo(self, percentage):
        whole_screen_size = self.screen.get_size()
        whole_screen_section =  RenderSectionHandler(whole_screen_size)
        web_bar_rectangle_parameters = [
            int(whole_screen_size[0]*0.7), 
            int(self.score_bar_size*0.1), 
            int(whole_screen_size[0]*0.2), 
            int(self.score_bar_size*0.8)
        ]
        web_bar_section = RenderSectionHandler(whole_screen_section, web_bar_rectangle_parameters)
        web_ammo_max_size = int(web_bar_section.get_rectangle_size()[0]*0.9)
        web_bar_length = int((1-percentage) * web_ammo_max_size)
        web_bar_ammo_section = RenderSectionHandler(web_bar_section, [
            #web_ammo_max_size - int(web_ammo_max_size * percentage), 
            web_bar_section.get_rectangle_size()[0] - web_bar_length,
            0, 
            web_bar_length, 
            int(web_bar_section.get_rectangle_size()[1])
        ])

        pygame.draw.rect(self.screen, (255, 255, 255), web_bar_section.get_rectangle_section_values())
        pygame.draw.rect(self.screen, (0, 0, 0), web_bar_ammo_section.get_rectangle_section_values())



    def __render_score_and_episode(self, score, high_score, game_info):
        pygame.draw.rect(self.screen, self.score_background_color, [0, 0, self.screen.get_width(), self.score_bar_size])
        self.__render_text("Score : ", score, (100, self.score_bar_size//4))
        self.__render_text("Highscore : ", high_score, (350, self.score_bar_size//4))
        self.__render_web_shooter_ammo(game_info['web_shooter_ammo_percentage'])


    def check_debug(self, camera, debug):
        if debug is None:
            return
        self.__render_object(debug, camera, self.debug_player_image)
    


    def render(self, render_dict : dict, score : int, high_score : int, debug=None):
        self.__check_for_events()
        self.screen.blit(self.background, self.background_tuple)
        self.__render_objects(render_dict)
        self.__render_score_and_episode(score, high_score, render_dict)
        self.check_debug(render_dict['camera'], debug)
        pygame.display.flip()
        # pygame.display.update()
        self.clock.tick(90)

        

