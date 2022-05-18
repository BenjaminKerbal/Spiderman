
import sys
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from Environment.Rendering.RenderingParent import RenderingParent
from Environment.Utils.ObjectShapes import ObjectShapes
from Environment.Utils.ObjectType import ObjectType
from Environment.Camera import Camera
from Environment.RenderSectionHandler import RenderSectionHandler

class LegacyRenderer(RenderingParent):

    def __init__(self, display_size, score_bar_size):
        pygame.init()
        self.display_size = display_size
        self.screen = pygame.display.set_mode(self.display_size)
        self.clock = pygame.time.Clock()
        self.score_bar_size = score_bar_size

        self.background_colour = (97, 147, 173)
        # self.player_colour = (0, 139, 194)
        self.debug_player_colour = (176, 79, 75)
        self.player_colour = (238, 24, 0)
        self.net_colour = (255, 255, 255)
        self.finish_colour = (255, 0, 0)
        self.roof_colour = (0, 0, 0)
        self.obstacle_colour = (100, 100, 100)
        self.score_background_color = (25, 25, 25)
        self.score_color = (200, 120, 230)
        self.score_font = pygame.font.SysFont('Comic Sans MS', 30)

    def __check_for_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def __get_pos_of_body(self, body, camera : Camera):
        positions = body.position
        return (int(positions[0] - camera.offset.x), int(positions[1]) + self.score_bar_size)

    def __compute_finish_line_pos(self, x_pos, camera):
        return (x_pos - camera.offset.x , -100), (x_pos - camera.offset.x , camera.DISPLAY_H + 100)

    def __render_object(self, object, camera : Camera, override_color=None):
        if override_color is not None:
            colour = override_color
        elif object['type'] == ObjectType.Player:
            colour = self.player_colour
        elif object['type'] == ObjectType.Roof:
            colour = self.roof_colour
        elif object['type'] == ObjectType.Obstacle:
            colour = self.obstacle_colour
        elif object['type'] == ObjectType.Net:
            colour = self.net_colour
        elif object['type'] == ObjectType.FinishLine:
            colour = self.finish_colour
        else:
            raise("Unrecognized type")
        
        if object['drawShape'] == ObjectShapes.Line:
            pos1 = self.__get_pos_of_body(object['shape'].a, camera)
            pos2 = self.__get_pos_of_body(object['shape'].b, camera)
            pygame.draw.line(self.screen, colour, pos1, pos2, object['size'])
        elif object['drawShape'] == ObjectShapes.FinishLine:
            pos1, pos2 = self.__compute_finish_line_pos(object['value'], camera)
            pygame.draw.line(self.screen, self.finish_colour, pos1, pos2, 5)
        else:
            pos = self.__get_pos_of_body(object['shape'].body, camera)
            if object['drawShape'] == ObjectShapes.Circle:
                pygame.draw.circle(self.screen, colour, pos, object['size'])
            elif object['drawShape'] == ObjectShapes.Rectangle:
                width = object['size'][0]
                height = object['size'][1]
                pos_x_start = pos[0] - width//2
                pos_y_start = pos[1] - height//2
                pygame.draw.rect(self.screen, colour, [pos_x_start, pos_y_start, width, height])
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

        '''
        if isinstance(game_info['debug_string'], list):
            start_y = 50
            for item in game_info['debug_string']:
                self.__render_text("Debug : ", episode, (500, self.score_bar_size//4))
        else:
            self.__render_text("Debug : ", episode, (500, self.score_bar_size//4))
        '''


    def check_debug(self, camera, debug):
        if debug is None:
            return
        self.__render_object(debug, camera, self.debug_player_colour)
        
        # for line in debug:
        #     colour = (100, 255, 0) if line['detected'] else (250, 100, 0)
        #     start = (line['start'][0], line['start'][1] + self.score_bar_size)
        #     target = (line['target'][0], line['target'][1] + self.score_bar_size)
        #     pygame.draw.line(self.screen, colour, start, target, 4)


    

    def render(self, render_dict : dict, score : int, high_score : int, debug=None):
        self.__check_for_events()
        self.screen.fill(self.background_colour)
        self.__render_objects(render_dict)
        self.__render_score_and_episode(score, high_score, render_dict)
        self.check_debug(render_dict['camera'], debug)
        pygame.display.update()
        self.clock.tick(90)

        

