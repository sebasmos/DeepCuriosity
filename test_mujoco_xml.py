import gymnasium as gym
from utils import set_seed, get_swimmer_xml

# make a simple name space to simulate the config

from types import SimpleNamespace
cfg = SimpleNamespace()
cfg.n_joints = 5
cfg.density = 3000
cfg.viscosity = 0.1
cfg.environment = 'Swimmer-v5'
xml_file = get_swimmer_xml(cfg)
env = gym.make(cfg.environment, xml_file=xml_file.as_posix())