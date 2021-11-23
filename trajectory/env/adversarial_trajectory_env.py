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
        self.start_collecting_rollout()


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


    def reset(self):
        self.create_simulation()
        self.lead_vehicle.reset()
        return self.get_state(_store_state=True)

    def step_adversary(self, action):
        print('STEPPING ADVERSARY')
        self.lead_vehicle.trajectory.append(action)
        done = False
        if len(self.lead_vehicle.trajectory) == self.max_steps:
            done = True
        return self.lead_vehicle.trajectory, 0, done, {}

    def reset_agent(self):
        print('collecting rollout')
        self.start_collecting_rollout()
        self.lead_vehicle.reset()
        return self.get_state(_store_state=True)

    def step(self, actions):
        # additional trajectory data that will be plotted in tensorboard
        metrics = {}

        # apply action to AV
        if self.av_controller == 'rl':
            if type(actions) not in [list, np.ndarray]:
                actions = [actions]
            for av, action in zip(self.avs, actions):
                accel = self.action_set[action] if self.discrete else float(action)
                metrics['rl_accel_before_failsafe'] = accel
                accel = av.set_accel(accel)  # returns accel with failsafes applied
                metrics['rl_accel_after_failsafe'] = accel
        elif self.av_controller == 'rl_fs':
            # RL with FS wrapper
            if type(actions) not in [list, np.ndarray]:
                actions = [actions]
            for av, action in zip(self.avs, actions):
                vdes_command = av.speed + float(action) * self.time_step
                metrics['rl_accel'] = float(action)
                metrics['vdes_command'] = vdes_command
                metrics['vdes_delta'] = float(action) * self.time_step
                av.set_vdes(vdes_command)  # set v_des = v_av + accel * dt

        # execute one simulation step
        end_of_horizon = not self.sim.step()

        # print progress every 5s if running from simulate.py
        if self.simulate and (end_of_horizon or time.time() - self.log_time_counter > 5.0):
            steps, max_steps = self.sim.step_counter, self.traj['size']
            print(f'Progress: {round(steps / max_steps * 100, 1)}% ({steps}/{max_steps} env steps)')
            self.log_time_counter = time.time()

        # compute reward & done
        h = self.avs[0].get_headway()
        th = self.avs[0].get_time_headway()

        reward = 0

        # prevent crashes
        crash = (h <= 0)
        if crash:
            reward -= 50.0

        # forcibly prevent the car from getting too small or large headways
        headway_penalties = {
            'low_headway_penalty': h < self.min_headway,
            'large_headway_penalty': h > self.max_headway,
            'low_time_headway_penalty': th < self.minimal_time_headway}
        if any(headway_penalties.values()):
            reward -= 2.0

        reward -= np.mean([max(self.sim.get_data(veh, 'instant_energy_consumption')[-1], 0) for veh in self.mpg_cars]) / 10.0
        if self.av_controller == 'rl':
            reward -= 0.002 * accel ** 2
        reward += 1

        # give average MPG reward at the end
        # if end_of_horizon:
        #     mpgs = [self.sim.get_data(veh, 'avg_mpg')[-1] for veh in self.mpg_cars]
        #     reward += np.mean(mpgs)

        # log some metrics
        metrics['crash'] = int(crash)
        for k, v in headway_penalties.items():
            metrics[k] = int(v)

        # get next state & done
        next_state = self.get_state(_store_state=True)
        done = (end_of_horizon or crash)
        infos = { 'metrics': metrics }

        if self.collect_rollout:
            self.collected_rollout['actions'].append(get_first_element(actions))
            self.collected_rollout['base_states'].append(self.get_base_state())
            self.collected_rollout['base_states_vf'].append(self.get_base_additional_vf_state())
            self.collected_rollout['rewards'].append(reward)
            self.collected_rollout['dones'].append(done)
            self.collected_rollout['infos'].append(infos)


        return next_state, reward, done, infos
