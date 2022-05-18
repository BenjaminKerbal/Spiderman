
''' Removes the headache of wanting to group multipe rendering parts and allows for easy editing '''

class RenderSectionHandler:

    def __init__(self, section, rectangle_parameters=None):
        if rectangle_parameters is None:
            self.section = RenderSectionHandler(section, [0, 0, section[0], section[1]])
            self.rectangle_parameters = [0, 0, section[0], section[1]]
            self.root = True
        else:
            self.section = section
            self.rectangle_parameters = rectangle_parameters
            self.root = False


    def get_rectangle_size(self):
        return (self.rectangle_parameters[2], self.rectangle_parameters[3])

    def get_pos(self, pos):
        this_section_pos = (self.rectangle_parameters[0] + pos[0], self.rectangle_parameters[1] + pos[1])
        if self.root:
            return this_section_pos
        else:
            return self.section.get_pos(this_section_pos)

    def get_rectangle_section_values(self):
        real_pos = self.get_pos((0, 0))
        return [
            real_pos[0], 
            real_pos[1],
            self.rectangle_parameters[2],
            self.rectangle_parameters[3]
        ]

    