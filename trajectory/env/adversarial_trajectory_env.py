import gym

from trajectory.env.trajectory_env import *

default_config = {
    'paired': True,
    'max_steps': 256,
    'no_eval': True,
    'av_controller': 'rl'
}

class AdversarialTrajectoryEnv(TrajectoryEnv):
    def __init__(self, seed=0, config={}):
        # TODO: Pass config into here
        default_config.update(config)
        config = default_config
        super().__init__(config)

        # This action space is really supposed to be for the protagonist agent, but it is the same for the protagonist, antagonist and adversary
        self.adversary_action_space = self.action_space
        self.adversary_ts_obs_space = gym.spaces.Box(
            low=0, high=config['max_steps'], shape=(1,), dtype='uint8')
        self.adversary_observation_space = {
            'time_step': self.adversary_ts_obs_space,
            'trajectory_obs': self.observation_space
        }
        self.reward_range = (-50.0, 50,0)

        self.lead_vehicle = None


    @property
    def processed_action_dim(self):
        return 1

    def create_simulation(self):
        # collect the next trajectory
        # PAIRED: Run adversarial vehicle model to get trajectory
            # Set self.traj; must have attributes 'timestep' and 'size'
            # TODO: figure out correct way to initialize. For now, it just needs to have a time step and size.
        self.traj = {'timestep': .05, 'size': 10}

            # if self.fixed_traj_path is not None and self.traj is None:
            #     self.traj = next(
            #         t for t in self.data_loader.trajectories
            #         if str(t['path']).split("/")[-1]
            #         == self.fixed_traj_path.split("/")[-1])
            #
            # if not self.fixed_traj_path:
            #     self.traj = next(self.trajectories)

        # create a simulation object
        self.time_step = self.traj['timestep']
        self.sim = Simulation(timestep=self.time_step)

        # populate simulation with a trajectory leader
        # PAIRED: Lead vehicle is the adversary
        self.sim.add_vehicle(controller='adversarial')
        self.lead_vehicle = self.sim.get_vehicles()[0]

        # parse platoons
        if self.platoon in PLATOON_PRESETS:
            print(f'Setting scenario preset "{self.platoon}"')
            self.platoon = PLATOON_PRESETS[self.platoon]

        # replace (subplatoon)*n into subplatoon ... subplatoon (n times)
        replace1 = lambda match: ' '.join([match.group(1)] * int(match.group(2)))
        self.platoon = re.sub(r'\(([a-z0-9\s\*\#]+)\)\*([0-9]+)', replace1, self.platoon)
        # parse veh#tag1...#tagk*n into (veh, [tag1, ..., tagk], n)
        self.platoon_lst = re.findall(r'([a-z]+)((?:\#[a-z]+)*)(?:\*?([0-9]+))?', self.platoon)

        # spawn vehicles
        self.avs = []
        self.humans = []
        for vtype, vtags, vcount in self.platoon_lst:
            for _ in range(int(vcount) if vcount else 1):
                tags = vtags.split('#')[1:]
                if vtype == 'av':
                    self.avs.append(
                        self.sim.add_vehicle(controller=self.av_controller, kind='av', tags=tags, gap=20, **eval(self.av_kwargs))
                    )
                elif vtype == 'human':
                    self.humans.append(
                        self.sim.add_vehicle(controller=self.human_controller, kind='human', tags=tags, gap=20, **eval(self.human_kwargs))
                    )
                else:
                    raise ValueError(f'Unknown vehicle type: {vtype}. Allowed types are "human" and "av".')

        # define which vehicles are used for the MPG reward
        self.mpg_cars = self.avs + (self.humans if self.include_idm_mpg else [])

        # initialize one data collection step
        self.sim.collect_data()

    def front_vehicle():
        followed_car = self.RL()

    def reset(self):
        self.create_simulation()
        self.lead_vehicle.reset()
        return self.get_state(_store_state=True)

    def step_adversary(self, action):
        self.lead_vehicle.trajectory.append(action)
