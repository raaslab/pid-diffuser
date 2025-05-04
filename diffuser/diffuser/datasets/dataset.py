import h5py
import numpy as np

class Dataset:
    def __init__(self, file_path):
        """
        Initializes the dataset.

        Args:
            file_path (str): Path to the .h5 or .hdf5 file.
            keys (list, optional): List of dataset keys to load (e.g., ["observations", "actions"]).
        """

        self.dataset = h5py.File(file_path, "r")
        self.actions = self.dataset["actions"][:]
        self.terminals = self.dataset["terminals"][:]
        self.timeouts = self.dataset["timeouts"][:]
        self.observations = self.dataset["observations"][:]
        self.rewards = self.dataset["rewards"][:]

        self.mapping = { 
            "actions": self.actions, 
            "terminals": self.terminals, 
            "timeouts": self.timeouts, 
            "observations": self.observations, 
            "rewards": self.rewards
        }
    
    def get(self, key):
        return self.mapping.get(key, None)



