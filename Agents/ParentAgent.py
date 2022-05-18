import io
import os
from os.path import join, exists, basename
import sys
import torch
import shutil
import json
from torchsummary import summary


''' The parent handler includes the nessesary function for all agent and will automatically save crusial 
information about the training and agent settings so it will be easier to check out an older model if wanted '''

class PostInitCaller(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__populate_results_folder__()
        return obj

class ParentAgent(metaclass=PostInitCaller):

    MODEL_FOLDER_NAME = "Models"
    BACKUP_PREFIX = "Backup_"
    EXTENSION = ".pth"
    AGENT_INFO = "agent_info.json"
    NETWORK_INFO = "network_info.txt"
    NETWORK_COPY = "network_copy.txt"
    AGENT_COPY = "agent_copy.txt"

    current_model_folder = None

    def store_transition(self, state, next_state, action, reward, done):
        raise Exception("Needs to be implemented!")

    def expects_one_action_value(self):
        raise Exception("Needs to be implemented!")

    def step_update(self):
        pass

    def end_of_episode_update(self):
        pass

    def __get_storage_information(self):
        module_location = self.__module__.split(".")
        general_model_folder = join(os.getcwd(), module_location[0], module_location[1], module_location[2], self.MODEL_FOLDER_NAME)
        if not exists(general_model_folder):
            os.makedirs(general_model_folder)
        agent_name = self.agent_name if  self.agent_name is not None else "Unnamed"
        current_model_folder = join(general_model_folder,agent_name)
        if not exists(current_model_folder):
            os.makedirs(current_model_folder)
        self.current_model_folder = current_model_folder
        return current_model_folder

    def __get_save_name(self, backup=False, extra_info=None):
        name = self.agent_name
        if not name.endswith(self.EXTENSION):
            name = name + self.EXTENSION
        name = name if extra_info is None else extra_info + "_" + name
        return self.BACKUP_PREFIX + name if backup else name

    def __save_model_info(self, model_folder):
        save_dict = {}
        agent_dict = self.__dict__
        for key, value in agent_dict.items():
            if isinstance(value, torch.Tensor):
                continue
            if key == "replay_memory" or key == "model_store_location":
                continue
            if isinstance(value, object):
                if hasattr(value, "__module__"):
                    save_dict[key] = value.__module__.split(".")[-1]
                    continue
            save_dict[key] = str(value)

        save_dict = json.dumps(save_dict, indent=2)
        with open(join(model_folder, self.AGENT_INFO), 'w') as fp:
            fp.write(save_dict)

        if "policy" in agent_dict:
            if "observation_space" in agent_dict:
                old_stdout = sys.stdout
                new_stdout = io.StringIO()
                sys.stdout = new_stdout
                observation_size = agent_dict["observation_space"]
                summary(agent_dict["policy"], (1, observation_size))
                model_structure = new_stdout.getvalue()
                sys.stdout = old_stdout
                with open(join(model_folder, self.NETWORK_INFO), "w") as text_file:
                    text_file.write(model_structure)
            else:
                print("Could not find observation space variable")
            
            policy_location = [os.getcwd()] + agent_dict["policy"].__module__.split(".")
            policy_location[-1] = policy_location[-1] + ".py"
            policy_location = join(*policy_location)
            shutil.copyfile(policy_location, join(model_folder, self.NETWORK_COPY))
        else:
            print("Policy variable could not be found")
        
        agent_location = [os.getcwd()] + self.__module__.split(".")
        agent_location[-1] = agent_location[-1] + ".py"
        agent_location = join(*agent_location)
        shutil.copyfile(agent_location, join(model_folder, self.AGENT_COPY))

    def __populate_results_folder__(self):
        if not self.eval_mode:
            model_folder = self.__get_storage_information()
            self.__save_model_info(model_folder)

    def save_model(self, backup=False, extra_info=None):
        save_name = self.__get_save_name(backup, extra_info)
        torch.save(self.policy.state_dict(), join(self.current_model_folder, save_name))
        print("Model saved in", basename(self.current_model_folder), "with name:", save_name)

    def load_model(self, name):
        module_location = self.__module__.split(".")
        general_model_folder = join(os.getcwd(), module_location[0], module_location[1], module_location[2], self.MODEL_FOLDER_NAME)
        folder_name = name[len(self.BACKUP_PREFIX):] if name.startswith(self.BACKUP_PREFIX) else name
        folder_name = folder_name[:-len(self.EXTENSION)] if folder_name.endswith(self.EXTENSION) else folder_name
        model_folder = join(general_model_folder, folder_name)
        if not exists(model_folder):
            print("Could not find model folder -", folder_name)
            return
        if not name.endswith(self.EXTENSION):
            name = name + self.EXTENSION
        file_path = join(model_folder, name)
        if not exists(file_path):
            print('Could not find model "{}" in folder {}'.format(name, model_folder))
            return
        self.policy.load_state_dict(torch.load(file_path, map_location="cpu"))
        if "target_net" in self.__dict__:
            self.target_net.load_state_dict(self.policy.state_dict())
            self.target_net.eval()
    


    
    

    
