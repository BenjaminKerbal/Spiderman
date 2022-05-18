from enum import Enum


class Agents(Enum):
    SimpleQ = "SimpleQ"
    DRQN = "DRQN"
    StateAgent = "StateAgent"
    ActorCritic = "ActorCritic"
    PolicyGradient = 'PolicyGradient'