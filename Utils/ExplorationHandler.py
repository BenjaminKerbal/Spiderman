from Utils.Enums.ExplorationTypes import ExplorationTypes

class ExplorationHandler:

    def __init__(self, exploration_type, max_episodes, train_mode, target_epsilon=None, target_tuning=False):
        self.max_episodes = max_episodes
        if not train_mode:
            self.epsilon_function = lambda x : 0
        elif exploration_type == ExplorationTypes.Gile:
            assert target_epsilon is not None, "Target param 1 cannot be None"
            self.glie_a = target_epsilon * max_episodes / (1 - target_epsilon)
            if target_tuning:
                self.epsilon_function = lambda x: target_epsilon
            else:
                self.epsilon_function = self.__get_gile_epsilon

        else:
            raise("Incorrect exploration type")
        
    def __get_gile_epsilon(self, episode):
        return self.glie_a / (self.glie_a + episode)

    def get_exploration(self, episode):
        return self.epsilon_function(episode)
