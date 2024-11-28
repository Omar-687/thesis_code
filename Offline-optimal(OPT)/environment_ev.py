import math
import random
from typing import Optional
import numpy as np
import gymnasium
from gymnasium import spaces
from scipy.stats import entropy
from index_based_algs import LeastLaxityFirstAlg
# import testing_days
from datetime import datetime, timedelta
from collections import deque  # Ordered collection with ends

import networks
from utils import *
class EVenvironment(gymnasium.Env):
    def __init__(self,
                 scheduling_algorithm,
                 cost_list,
                 max_charging_rate,
                 tuning_parameter,
                 train=True,
                 evse=54,
                 power_levels=10,
                 time_between_timesteps=12,
                 power_limit=150,
                 charging_days_per_charging_station=None,
                 charging_stations=None,
                 load_data_via_json=True,
                 data_files = None,
                 # o1=0.1,
                 o1=0.1,
                 o2=0.2,
                 # o2=0.2,
                 # o2=0.2,
                 # o1=0.1,
                 # o2=0.2,
                 o3=2,
                 costs_in_kwh=False
                 # o3=0.2
                 ):
        self.number_of_additional_observation_parameters = 2
        self.scheduling_algorithm = scheduling_algorithm
        low_vector = np.zeros(shape=(evse*2+self.number_of_additional_observation_parameters,))
        high_vector = np.zeros(shape=(evse*2+self.number_of_additional_observation_parameters,))
        self.evse = evse
        self.aggregator_state = np.zeros(self.evse * 2 + self.number_of_additional_observation_parameters)
        self.dict_arrivals_departures = {}
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


        self.entropies_for_each_step = []
        self.chosen_day_index = 0
        self.charging_stations = charging_stations
        self.activated_evse = np.zeros(shape=(self.evse,))
        self.evse_map_to_ev = np.zeros(shape=(self.evse,)) - 1
        self.charging_days_per_charging_station = charging_days_per_charging_station
        self.power_levels = power_levels
        self.costs_list = cost_list
        self.time_between_timesteps = time_between_timesteps
        self.power_limit = power_limit
        self.delta = (self.time_between_timesteps/60)
        # self.signal_buffer = deque(maxlen=3)
        self.signal_buffer = deque(maxlen=1)
        self.max_charging_rate = max_charging_rate
        self.tuning_parameter = tuning_parameter
        self.timestep = 0
        # not necessary because max charging rate is the same for all EVs
        # self.map_evse_to_ev = []
        self.load_data_via_json = load_data_via_json
        if self.load_data_via_json:
            self.data_files = data_files
        self.train = train
        self.o1 = o1
        self.o2 = o2
        self.o3 = o3
        # add signal to the observation space
        self.signal_ut = 0
        self.cumulative_costs = 0
        self.max_timestep =  int((60 * 24) / self.time_between_timesteps) - 1
        self.smoothing = True
        self.current_time = 0
        self.smoothing_coeff = 0
        self.max_ramp_rate = 30
        # self.smoothing_coeff = 0
        self.normalised_pts = []
        self.costs_per_u = []
        self.optim_results_per_u = []
        self.not_fully_charged_before_departure_penalties = []
        self.aggregator_states_for_each_timestep = []
        self.chosen_ut_for_each_timestep = []
        self.chosen_sum_of_charging_rates = []
        self.charging_rates_matrix = np.array([])
        # tuples (index of ev, undelivered energy) delivered energy we will find in outside function
        self.delivered_and_undelivered_energy = {}
        self.costs_in_kwh = costs_in_kwh

    def map_to_interval(self,num, interval):
        lower, upper = interval
        return max(lower, min(num, upper))

    def ppc(self, c_t, p_t):
        costs_per_u = [0.0 for i in range(self.power_levels)]
        optim_results_per_u = [0.0 for i in range(self.power_levels)]
        previous_u_t = np.mean(self.signal_buffer)
        low = max(previous_u_t - self.max_ramp_rate, 0)
        high = min(previous_u_t + self.max_ramp_rate,self.power_limit)
        # generate static power signals
        u_ts = np.linspace(low, high, num=self.power_levels )
        # u_ts[1] = self.max_charging_rate
        min_value = np.inf
        min_value_index = 0
        for i in range(self.power_levels):
            costs = c_t * (u_ts[i] * self.delta)
            costs_per_u[i] = costs
            if p_t[i] == 0:
                optim_results_per_u[i] = float('nan')
                continue

            # np log computes log of base e
            value = costs - self.tuning_parameter * np.log(p_t[i])
            optim_results_per_u[i] = value
            if value < min_value:
                min_value = value
                min_value_index = i
        if not self.train:
            self.normalised_pts.append(p_t)
            self.costs_per_u.append(costs_per_u)
            self.optim_results_per_u.append(optim_results_per_u)

        chosen_ut_index = min_value_index

        res = u_ts[chosen_ut_index]
        res = ((self.smoothing_coeff)*np.mean(self.signal_buffer)) + ((1 - self.smoothing_coeff)*res)
        # else:
        #     res = 0
        self.signal_buffer.append(res)
        # if self.smoothing:
        #     res = self.smoothing_coeff * np.mean(self.signal_buffer) + (1 - self.smoothing_coeff) * res
        #     self.signal_buffer.append(res)

        return res

    # if costs were real time, at each timestep they would be loaded
    # assume that we normalise the probabilistic actions
    def step(self, action):
        max_timestep = int((60 * 24) / self.time_between_timesteps) - 1
        self.timestep += 1
        self.current_time += self.delta
        self.aggregator_state[-1] += self.delta
        current_time = self.current_time
        # clipping action into MEF space
        action = action / np.sum(action)
        if self.timestep == max_timestep:
            action =  [0 for i in range(self.power_levels)]
            action[0] = 1
            self.signal_ut = self.ppc(c_t=self.costs_list[self.timestep],
                                      p_t=action)
        if self.timestep < max_timestep:
            self.signal_ut = self.ppc(c_t=self.costs_list[self.timestep],
                                      p_t=action)


        waiting_evs_to_connect = []
        # collecting cars which currently arrived
        for ev in self.charging_data:
            index, arrival, departure, maximum_charging_rate, requested_energy = ev
            departure_not_normalised_time = convert_timestep_to_hours(timestep=departure,
                                                                      time_between_timesteps=self.time_between_timesteps)
            # checking arrivals for x_{t + 1}
            if self.timestep == arrival + 1:
                # keep not normalised time, because if period changes, you would need to set everytime new limits
                waiting_evs_to_connect.append([index,
                                               departure_not_normalised_time - current_time,
                                               requested_energy])

                if not self.train:
                    self.delivered_and_undelivered_energy[index] = [requested_energy]
                    self.dict_arrivals_departures[index] = [arrival, departure]
        ev_index = 0
        for evse_index in range(self.evse):
            map_evse_to_obs = evse_index * 2
            # if there are no more cars waiting to connect
            if ev_index == len(waiting_evs_to_connect):
                break
            # if evse is already taken
            if self.activated_evse[evse_index] == 1:
                continue

            ev_identifier, remaining_charging_time, requested_energy = waiting_evs_to_connect[ev_index]
            # assign ev to evse
            self.activated_evse[evse_index] = 1
            self.aggregator_state[map_evse_to_obs] = remaining_charging_time
            self.aggregator_state[map_evse_to_obs + 1] = requested_energy
            self.evse_map_to_ev[evse_index] = ev_identifier

            ev_index += 1
        # check if there werent more evs than evse
        if ev_index != len(waiting_evs_to_connect):
            raise ValueError('Not enough EVSE to charge EVs')

        # change algorithm to generic after fixing errors
        # decrease accuracy if you can to have faster algorithm
        sch_alg = LeastLaxityFirstAlg(EVs=self.charging_data,
                             start=self.charging_date,
                             end=self.end_charging_date,
                             time_between_timesteps=self.time_between_timesteps,
                             accuracy=1e-4,
                             number_of_evse=self.evse,
                             process_output=True,
                             costs_loaded_manually=[self.costs_list[self.timestep]],
                             # for improving time spent, we dont create arrays from time horizon available energy at each timestep and future info about costs
                             )


        schedule = sch_alg.solve_for_current_timestep_given_observation(observation=self.aggregator_state,
                                                                        maximum_charging_rate=self.max_charging_rate*self.delta,
                                                                        available_energy=self.signal_ut*self.delta,
                                                                        number_of_evse=self.evse,
                                                                        activated_evse=self.activated_evse)
        schedule = np.array(schedule)
        schedule_in_kw = schedule / self.delta
        not_fully_charged_until_departure_penalty = 0
        for i in range(self.evse):
            ev_map_to_obs = i * 2
            if self.activated_evse[i] == 0:
                continue
            given_energy_to_ev_on_evse = schedule[i]
            self.aggregator_state[ev_map_to_obs] -= self.delta
            delivered_energy = min(given_energy_to_ev_on_evse,
                                   self.aggregator_state[ev_map_to_obs + 1])
            self.aggregator_state[ev_map_to_obs + 1] -=  delivered_energy
            if not self.train:
                self.charging_rates_matrix[int(self.evse_map_to_ev[i])][int(self.timestep)] = delivered_energy

            # self.aggregator_state[ev_map_to_obs] == 0
            if self.aggregator_state[ev_map_to_obs] < self.delta and self.activated_evse[i] == 1:
                undelivered_energy = self.aggregator_state[ev_map_to_obs + 1]
                not_fully_charged_until_departure_penalty += undelivered_energy
                index_of_ev = self.evse_map_to_ev[i]
                if not self.train:
                    self.delivered_and_undelivered_energy[index_of_ev].append(undelivered_energy)

                self.evse_map_to_ev[i] = -1
                self.aggregator_state[ev_map_to_obs + 1] = 0
                self.aggregator_state[ev_map_to_obs] = 0
                self.activated_evse[i] = 0
                # if self.timestep == max_timestep and self.activated_evse[i] == 1 and not self.train:
                #     undelivered_energy = self.aggregator_state[ev_map_to_obs + 1]
                #     index_of_ev = self.evse_map_to_ev[i]
                #     self.delivered_and_undelivered_energy[index_of_ev].append(undelivered_energy)

        third_term = self.o3 * abs((self.signal_ut*self.delta) - math.fsum(schedule))
        # we include in schedule only currently charged evs
        reward = (entropy(action) + self.o1*math.fsum(schedule)
                  - self.o2 * not_fully_charged_until_departure_penalty - third_term)
        # third part is important else there is not much to learn if ut is totally random
        # the agent will not learn if ut is random bc it will satisfy its problems anyway for many different solutions
        reward = float(reward)

        # if self.smoothing:
        #     self.signal_ut = self.smoothing_coeff * self.aggregator_state[-2] + (1 - self.smoothing) * self.signal_ut

        self.aggregator_state[-2] = self.signal_ut

        # reward = entropy(scaled_action) + self.o1 * charging_performance_reward1 - self.o2
        terminated = False

        if self.timestep + 1 > max_timestep:
            terminated = True
        self.chosen_ut_for_each_timestep.append(self.signal_ut)
        if not self.train:
            self.entropies_for_each_step.append(entropy(action))
                # self.cumulative_costs += self.costs_list[self.timestep] * self.signal_ut
            self.chosen_sum_of_charging_rates.append(math.fsum(schedule))
            self.not_fully_charged_before_departure_penalties.append(not_fully_charged_until_departure_penalty)
            if not terminated:
                self.aggregator_states_for_each_timestep.append(deepcopy(self.aggregator_state))
        info = {}
        truncated = False


        return self.aggregator_state, reward, terminated, truncated, info

    # test this once you are sure what data you will use
    def reset(self, seed=None, options=None):
        self.aggregator_state = np.zeros(self.evse*2 +
                                         self.number_of_additional_observation_parameters,
                                         dtype=np.float64)
        time_to_add = timedelta(hours=23, minutes=59, seconds=59)
        # during training load from json file, it is easier than through timeseries and takes less time

        if  options is None:
            # we need to input correct charging days to make it work
            random_charging_station_index = random.randint(0, len(self.charging_stations) - 1)
            list_of_keys = self.charging_days_per_charging_station[random_charging_station_index]

            # self.chosen_day_index = self.chosen_day_index %
            sampled_day = np.random.choice(list_of_keys,size=1,replace=False)[0]
            self.charging_date = sampled_day
            self.end_charging_date = self.charging_date + time_to_add

            evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = {}, {}, {}
            number_of_evs_interval = [30, np.inf]
            if not self.load_data_via_json:
                evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
                    charging_network=networks.charging_networks[0],
                    # garages=caltech_garages,
                    # garages=networks.caltech_garages[:1],
                    garages=networks.caltech_garages,
                    start=self.charging_date,
                    end=self.end_charging_date,
                    period=self.time_between_timesteps,
                    max_charging_rate_within_interval=[self.max_charging_rate, self.max_charging_rate],
                    number_of_evs_interval=number_of_evs_interval,
                    include_weekends=False,
                    include_overday_charging=False
                )
            else:
                evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = get_evs_data_from_document_advanced_settings(
                    document=self.data_files[random_charging_station_index],
                    start=self.charging_date,
                    end=self.end_charging_date,
                    period=self.time_between_timesteps,
                    max_charging_rate_within_interval=[self.max_charging_rate, self.max_charging_rate],
                    number_of_evs_interval=number_of_evs_interval,
                    include_weekends=False,
                    allow_overday_charging=False,
                    dates_in_ios_format=True
                )
            self.charging_data = evs_timestamp_reset[self.charging_date]

        # basic testing
        elif options is not None and ('charging_data' in options.keys()) :
            self.charging_data = options['charging_data']
            # default charging date so we wont get errors
            self.charging_date = datetime(2020,1,1,0,0,0)
            self.end_charging_date = datetime(2020, 1, 1, 23, 59, 59)

        # testing
        elif options is not None and ('chosen_day' in options.keys()):
            self.charging_date = options['chosen_day']
            self.end_charging_date = self.charging_date + time_to_add
            chosen_network = networks.charging_networks[0]
            chosen_garages = networks.caltech_garages

            if 'charging_network' in options.keys() and 'garages' in options.keys():
                chosen_network = options['charging_network']
                chosen_garages = options['garages']
            else:
                ...
            evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = {}, {}, {}
            number_of_evs_interval = [30, np.inf]
            if 'document' in options.keys():
                evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = get_evs_data_from_document_advanced_settings(
                    document=options['document'],
                    start=self.charging_date,
                    end=self.end_charging_date,
                    number_of_evs_interval=number_of_evs_interval,
                    include_weekends=False,
                    allow_overday_charging=False,
                    period=self.time_between_timesteps,
                    dates_in_ios_format=True)
            else:
                evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
                    charging_network=chosen_network,
                    # garages=networks.caltech_garages[:1],
                    garages=chosen_garages,
                    start=self.charging_date,
                    end=self.end_charging_date,
                    period=self.time_between_timesteps,
                    max_charging_rate_within_interval=[self.max_charging_rate, self.max_charging_rate],
                    number_of_evs_interval=number_of_evs_interval,
                    include_weekends=False,
                    include_overday_charging=False
                )
            self.charging_data = evs_timestamp_reset[self.charging_date]


        # starting with t=0
        self.timestep = 0
        # initalise starting pt
        self.current_time = 0
        self.activated_evse = np.zeros(shape=(self.evse,))
        self.evse_map_to_ev = np.zeros(shape=(self.evse,)) - 1
        # result/debugging vars
        self.chosen_sum_of_charging_rates = []
        self.chosen_ut_for_each_timestep = []
        self.delivered_and_undelivered_energy = {}
        self.costs_per_u = []
        self.normalised_pts = []
        self.optim_results_per_u = []
        self.aggregator_states_for_each_timestep = []
        self.dict_arrivals_departures = {}
        self.not_fully_charged_before_departure_penalties = []
        self.signal_buffer.clear()
        self.signal_buffer.append(0)
        p_t = np.zeros(shape=(self.power_levels,))
        p_t[0] = 1
        self.entropies_for_each_step = [entropy(p_t)]
        self.signal_ut = self.ppc(c_t=self.costs_list[self.timestep],p_t=p_t)
        self.aggregator_state[-2] = self.signal_ut
        num_of_timesteps = int((60 * 24) / self.time_between_timesteps)
        self.charging_rates_matrix = np.zeros(shape=(len(self.charging_data),
                                                     num_of_timesteps))
        # add checking for arrivals in reset but maybe it doesnt matter because it is short time period
        if not self.train:
            self.aggregator_states_for_each_timestep.append(deepcopy(self.aggregator_state))

            self.aggregator_state[-1] += self.delta
            self.aggregator_states_for_each_timestep.append(deepcopy(self.aggregator_state))
            self.chosen_ut_for_each_timestep.append(self.signal_ut)
            # chosen signal ut at start
            self.chosen_sum_of_charging_rates.append(0)
        info = {}
        return self.aggregator_state, info


# Register the environment
gymnasium.envs.registration.register(
    id='EVenvironment-v0',
    entry_point='environment_ev:EVenvironment',  # Adjust the path as necessary
)


