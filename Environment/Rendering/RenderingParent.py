
from abc import ABC, abstractmethod



class RenderingParent(ABC):

    @abstractmethod
    def render(self, render_dict : dict, score : int, high_score : int, debug=None):
        pass


    
    

    
