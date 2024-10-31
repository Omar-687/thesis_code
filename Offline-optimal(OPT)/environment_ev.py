from typing import Optional
import numpy as np
import gymnasium
from gymnasium import spaces
from scipy.stats import entropy
# import testing_days
from datetime import datetime, timedelta
import networks
from utils import *
class EVenvironment(gymnasium.Env):

    def __init__(self,
                 scheduling_algorithm,
                 charging_days_list,
                 cost_list,
                 train=True,
                 evse=54,
                 tuning_parameter=6e3,
                 max_charging_rate=6.6,
                 power_levels=10,
                 time_between_timesteps=12,
                 power_rating=150,
                 training_episodes=500,
                 o1=0.1,
                 o2=0.2,
                 o3=2):
        self.number_of_additional_observation_parameters = 2
        self.scheduling_algorithm = scheduling_algorithm
        low_vector = np.zeros(shape=(evse*2+self.number_of_additional_observation_parameters,))
        high_vector = np.zeros(shape=(evse*2+self.number_of_additional_observation_parameters,))
        self.evse = evse
        self.aggregator_state = np.zeros(self.evse * 2 + 1)

        for i in range(self.evse):
            high_vector[i*2] = 24
            # lets say this value can be infinite but we have to set limit for observation
            high_vector[(i * 2) + 1] = 100
        high_vector[-1] = 24
        high_vector[-2] = 150
        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        # Box - Supports continuous (and discrete) vectors or matrices, used for vector observations

        self.observation_space = spaces.Box(low=low_vector, high=high_vector, dtype=np.float64)
        low_bound_action_space = np.zeros(shape=(power_levels,))
        upper_bound_action_space = np.ones(shape=(power_levels,))
        self.action_space = spaces.Box(low=low_bound_action_space, high=upper_bound_action_space)

        self.activated_evse = np.zeros(shape=(self.evse,))

        self.evse_map_to_ev = np.zeros(shape=(self.evse,))
        self.charging_days_list = charging_days_list
        self.power_levels = power_levels
        self.costs_list = cost_list
        self.time_between_timesteps = time_between_timesteps
        self.power_rating = power_rating
        self.delta = (self.time_between_timesteps/60)
        self.max_charging_rate = max_charging_rate
        self.tuning_parameter = tuning_parameter
        self.charging_days = charging_days_list
        self.timestep = 0
        # not necessary because max charging rate is the same for all EVs
        # self.map_evse_to_ev = []
        self.train = train
        self.o1 = 0.1
        self.o2 = 0.2
        self.o3 = 2
        # add signal to the observation space
        self.signal_ut = 0
        self.chosen_ut_for_each_timestep = []
        self.chosen_sum_of_charging_rates = []
        # tuples (index of ev, undelivered energy) delivered energy we will find in outside function
        self.delivered_and_undelivered_energy = {}
    def ppc(self, c_t, p_t):
        low = 0
        high = self.power_rating
        # generate static power signals
        u_ts = np.linspace(low, high, num=self.power_levels)
        min_value = np.inf
        min_value_index = 0
        for i in range(self.power_levels):
            if p_t[i] == 1:
                min_value_index = i
                break
            # action generated with such flexibility doesnt guarantee feasibility
            if p_t[i] == 0:
                continue
            # np log computes log of base e
            value = (c_t * u_ts[i]) - self.tuning_parameter * np.log(p_t[i])
            if value < min_value:
                min_value = value
                min_value_index = i
        chosen_ut_index = min_value_index
        return u_ts[chosen_ut_index]

    # def find_free_evse(self):
    #     for i in range(0,len(self.aggregator_state) - 1, 2):
    #         if self.aggregator_state[i] == 0:
    #             return i
    #     return None
    # if costs were real time, at each timestep they would be loaded
    # assume that we normalise the probabilistic actions
    def step(self, action):
        self.timestep += 1
        self.aggregator_state[-1] += self.delta
        current_time = self.aggregator_state[-1]
        # clipping action into MEF space
        action = action / np.sum(action)
        waiting_evs_to_connect = []
        # collecting cars which currently arrived
        for ev in self.charging_data:
            index, arrival, departure, maximum_charging_rate, requested_energy = ev
            departure_not_normalised_time = convert_timestep_to_hours(timestep=departure,
                                                                      time_between_timesteps=self.time_between_timesteps)
            if self.timestep == arrival:
                # keep not normalised time, because if period changes, you would need to set everytime new limits
                waiting_evs_to_connect.append([index,
                                               departure_not_normalised_time - current_time,
                                               requested_energy])
                if not self.train:
                    self.delivered_and_undelivered_energy[index] = [requested_energy]
        ev_index = 0
        taken_evse = 0
        for evse_index in range(self.evse):
            map_evse_to_obs = evse_index * 2
            # if there are no more cars waiting to connect
            if ev_index == len(waiting_evs_to_connect):
                break
            # if evse is already taken based on remaining charging time > 0
            if self.aggregator_state[map_evse_to_obs] > 0:
                taken_evse += 1
                continue

            ev_identifier, remaining_charging_time, requested_energy = waiting_evs_to_connect[ev_index]
            # assign ev to evse
            self.aggregator_state[map_evse_to_obs] = remaining_charging_time
            self.aggregator_state[map_evse_to_obs + 1] = requested_energy
            self.evse_map_to_ev[evse_index] = ev_identifier
            taken_evse += 1
            ev_index += 1
        # check if there werent more evs than evse
        if ev_index != len(waiting_evs_to_connect):
            raise ValueError('Not enough EVSE to charge EVs')

        # decrease accuracy if you can to have faster algorithm
        sch_alg = self.scheduling_algorithm(EVs=self.charging_data,
                             start=self.charging_date,
                             end=self.end_charging_date,
                             available_energy_for_each_timestep=None,
                             time_between_timesteps=self.time_between_timesteps,
                             accuracy=1e-4,
                             number_of_evse=self.evse,
                             cost_function=None,
                             process_output=True,
                             costs_loaded_manually=[self.costs_list[self.timestep]],
                             # for improving time spent, we dont create arrays from time horizon available energy at each timestep and future info about costs
                             info_about_future_costs=False,
                             set_available_energy_for_each_timestep=False,
                             set_time_horizon=False)

        # get schedule given current state x_t and current signal u_t
        schedule = sch_alg.solve_for_current_timestep_given_observation(observation=self.aggregator_state,
                                                     maximum_charging_rate=self.max_charging_rate,
                                                     available_energy=self.signal_ut,
                                                     number_of_evse=self.evse)
        not_fully_charged_until_departure_penalty = 0
        for i in range(self.evse):
            ev_map_to_obs = i * 2
            # disconnect evs from evse
            if self.aggregator_state[ev_map_to_obs] <= 0:
                undelivered_energy = self.aggregator_state[ev_map_to_obs + 1]
                not_fully_charged_until_departure_penalty += undelivered_energy
                index_of_ev = self.evse_map_to_ev[i]
                if not self.train:
                    self.delivered_and_undelivered_energy[index_of_ev].append(undelivered_energy)
                self.evse_map_to_ev[i] = 0
                self.aggregator_state[ev_map_to_obs + 1] = 0
                self.aggregator_state[ev_map_to_obs] = 0
            else:
                # update remaining charging time and remaining energy to be charged
                given_energy_to_ev_on_evse = schedule[i]
                self.aggregator_state[ev_map_to_obs] -= self.delta
                self.aggregator_state[ev_map_to_obs + 1] -=  min(given_energy_to_ev_on_evse,
                                                                 self.aggregator_state[ev_map_to_obs + 1])

        reward = np.sum(schedule)  - 0.5 * not_fully_charged_until_departure_penalty
        reward = float(reward)
        # technically we can load costs even in real time in testing
        self.signal_ut = self.ppc(c_t=self.costs_list[self.timestep],
                                  p_t=action)
        self.aggregator_state[-2] = self.signal_ut

        # reward = entropy(scaled_action) + self.o1 * charging_performance_reward1 - self.o2
        terminated = False
        max_timestep =  int((60*24)/self.time_between_timesteps) - 1
        if self.timestep >= max_timestep:
            terminated = True
        if not self.train:
            if not terminated:
                self.chosen_ut_for_each_timestep.append(self.signal_ut)

            self.chosen_sum_of_charging_rates.append(math.fsum(schedule))
        info = {}
        truncated = False
        return self.aggregator_state, reward, terminated, truncated, info

    # test this once you are sure what data you will use
    def reset(self, seed=None, options=None):
        self.aggregator_state = np.zeros(self.evse*2 +
                                         self.number_of_additional_observation_parameters,
                                         dtype=np.float64)
        time_to_add = timedelta(hours=23, minutes=59, seconds=59)

        if  options is None:
            # we need to input correct charging days to make it work
            list_of_keys = self.charging_days_list
            sampled_day = np.random.choice(list_of_keys,size=1,replace=False)[0]
            self.charging_date = sampled_day
            self.end_charging_date = self.charging_date + time_to_add

            number_of_evs_interval = [30, np.inf]
            evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
                charging_network=networks.charging_networks[0],
                # garages=caltech_garages,
                garages=networks.caltech_garages[:1],
                start=self.charging_date,
                end=self.end_charging_date,
                period=self.time_between_timesteps,
                max_charging_rate_within_interval=[self.max_charging_rate, self.max_charging_rate],
                number_of_evs_interval=number_of_evs_interval,
                include_weekends=False,
            )
            self.charging_data = evs_timestamp_reset[self.charging_date]

        elif options is not None and ('charging_data' in options.keys()) :
            self.charging_data = options['charging_data']
            # default charging date so we wont get errors
            self.charging_date = datetime(2020,1,1,0,0,0)
            self.end_charging_date = datetime(2020, 1, 1, 23, 59, 59)

        elif options is not None and ('chosen_day' in options.keys()):
            self.charging_date = options['chosen_day']
            self.end_charging_date = self.charging_date + time_to_add

            number_of_evs_interval = [30, np.inf]
            evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
                charging_network=networks.charging_networks[0],
                garages=networks.caltech_garages[:1],
                start=self.charging_date,
                end=self.end_charging_date,
                period=self.time_between_timesteps,
                max_charging_rate_within_interval=[self.max_charging_rate, self.max_charging_rate],
                number_of_evs_interval=number_of_evs_interval,
                include_weekends=False,
            )
            self.charging_data = evs_timestamp_reset[self.charging_date]



        self.timestep = 0
        p_t = np.zeros(shape=(self.power_levels,))
        p_t[0] = 1
        self.signal_ut = self.ppc(c_t=self.costs_list[self.timestep],p_t=p_t)
        self.aggregator_state[-2] = self.signal_ut
        if not self.train:
            self.chosen_ut_for_each_timestep.append(self.signal_ut)
        info = {}
        return self.aggregator_state, info


# Register the environment
gymnasium.envs.registration.register(
    id='EVenvironment-v0',
    entry_point='environment_ev:EVenvironment',  # Adjust the path as necessary
)


