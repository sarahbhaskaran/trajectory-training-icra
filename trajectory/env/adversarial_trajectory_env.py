from trajectory.env.trajectory_env import TrajectoryEnv

default_config = {
    'paired': True
}

class AdversarialTrajectoryEnv(TrajectoryEnv):
    def __init__(self, seed=0, config={}):
        # TODO: Pass config into here
        if config:
            super().__init__(config)
        else:
            super().__init__(default_config)

        # This action space is really supposed to be for the protagonist agent, but it is the same for the protagonist, antagonist and adversary
        self.adversary_action_space = self.action_space

    @property
    def processed_action_dim(self):
        return 1

    def front_vehicle():
        followed_car = self.RL()
