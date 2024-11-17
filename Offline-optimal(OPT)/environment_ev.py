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
                 charging_days_list,
                 cost_list,
                 max_charging_rate,
                 tuning_parameter,
                 train=True,
                 evse=54,
                 power_levels=10,
                 time_between_timesteps=12,
                 power_limit=150,
                 training_episodes=500,
                 limit_ramp_rates=True,
                 o1=1.3,
                 o2=1,
                 # o2=0.2,
                 # o1=0.1,
                 # o2=0.2,
                 o3=2,
                 costs_in_mwh=False
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

        self.chosen_day_index = 0
        self.activated_evse = np.zeros(shape=(self.evse,))
        self.evse_map_to_ev = np.zeros(shape=(self.evse,)) - 1
        self.charging_days_list = charging_days_list
        self.power_levels = power_levels
        self.costs_list = cost_list
        self.time_between_timesteps = time_between_timesteps
        self.power_limit = power_limit
        self.delta = (self.time_between_timesteps/60)
        # self.signal_buffer = deque(maxlen=3)
        self.signal_buffer = deque(maxlen=5)
        self.max_charging_rate = max_charging_rate
        self.tuning_parameter = tuning_parameter
        self.charging_days = charging_days_list
        self.timestep = 0
        # not necessary because max charging rate is the same for all EVs
        # self.map_evse_to_ev = []
        self.limit_ramp_rates = limit_ramp_rates
        self.train = train
        self.o1 = o1
        self.o2 = o2
        self.o3 = o3
        # add signal to the observation space
        self.signal_ut = 0
        self.cumulative_costs = 0
        self.smoothing = True
        # possibly too small coefficient
        # self.smoothing_coeff = 0.35
        # self.smoothing_coeff = 0.5
        # self.smoothing_coeff = 0.45
        # self.smoothing_coeff = 0.35
        self.smoothing_coeff = 0.6
        self.chosen_ut_for_each_timestep = []
        self.chosen_sum_of_charging_rates = []
        self.charging_rates_matrix = np.array([])
        # tuples (index of ev, undelivered energy) delivered energy we will find in outside function
        self.delivered_and_undelivered_energy = {}
        self.costs_in_mwh = costs_in_mwh

    def map_to_interval(self,num, interval):
        lower, upper = interval
        return max(lower, min(num, upper))

    def ppc(self, c_t, p_t):
        low = 0
        high = self.power_limit
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
        res = u_ts[chosen_ut_index]
        # res = random.choice(u_ts)
        # doesnt work well with limit ramp rates, possibly it limits the system operator to schedule
        # if self.limit_ramp_rates and abs(self.signal_ut - u_ts[chosen_ut_index]) > (U_diameter/2):
        #     res = self.signal_ut + self.map_to_interval(self.signal_ut - u_ts[chosen_ut_index],
        #                          [-(U_diameter/2), (U_diameter/2)])
        # if self.smoothing:
        #     res = self.smoothing_coeff * np.mean(self.signal_buffer) + (1 - self.smoothing_coeff) * res
        #     self.signal_buffer.append(res)

        return res

    # if costs were real time, at each timestep they would be loaded
    # assume that we normalise the probabilistic actions
    def step(self, action):
        max_timestep = int((60 * 24) / self.time_between_timesteps) - 1
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
            # disconnect evs from evse
            # at time d(i) evs will get charging rate 0
            # change to charging [a(i),d(i)]
            if self.aggregator_state[ev_map_to_obs] == 0 and self.activated_evse[i] == 1:
                undelivered_energy = self.aggregator_state[ev_map_to_obs + 1]
                not_fully_charged_until_departure_penalty += undelivered_energy
                index_of_ev = self.evse_map_to_ev[i]
                if not self.train:
                    self.delivered_and_undelivered_energy[index_of_ev].append(undelivered_energy)

                self.evse_map_to_ev[i] = -1
                self.aggregator_state[ev_map_to_obs + 1] = 0
                self.aggregator_state[ev_map_to_obs] = 0
                self.activated_evse[i] = 0
            else:
                # covers two cases
                # 1. when evse are not active, in that case we dont update anything
                # 2. when evse are active and evs have not finished charging - in that case we update state
                # update remaining charging time and remaining energy to be charged
                given_energy_to_ev_on_evse = schedule[i]
                self.aggregator_state[ev_map_to_obs] -= min(self.delta,
                                                            self.aggregator_state[ev_map_to_obs])
                delivered_energy = min(given_energy_to_ev_on_evse,
                                       self.aggregator_state[ev_map_to_obs + 1])
                self.aggregator_state[ev_map_to_obs + 1] -=  delivered_energy
                if not self.train:
                    self.charging_rates_matrix[int(self.evse_map_to_ev[i])][int(self.timestep)] = delivered_energy

                if self.timestep >= max_timestep and self.activated_evse[i] == 1 and not self.train:
                    undelivered_energy = self.aggregator_state[ev_map_to_obs + 1]
                    index_of_ev = self.evse_map_to_ev[i]
                    self.delivered_and_undelivered_energy[index_of_ev].append(undelivered_energy)

        # normal_3rdterm = self.o3*abs(self.signal_ut - math.fsum(schedule))**2/self.power_rating
        # mse = self.o3*abs(self.signal_ut - math.fsum(schedule_in_kw))**2/self.power_limit
        third_term = self.o3 * abs(self.signal_ut - math.fsum(schedule_in_kw))
        # mse = self.o3 * abs(self.signal_ut - math.fsum(schedule))
        # we include in schedule only currently charged evs
        reward = (entropy(action) + self.o1*math.fsum(schedule_in_kw)
                  - self.o2 * not_fully_charged_until_departure_penalty - third_term)
        # third part is important else there is not much to learn if ut is totally random
        # the agent will not learn if ut is random bc it will satisfy its problems anyway for many different solutions
        reward = float(reward)
        # technically we can load costs even in real time in testing
        self.signal_ut = self.ppc(c_t=self.costs_list[self.timestep],
                                  p_t=action)
        # if self.smoothing:
        #     self.signal_ut = self.smoothing_coeff * self.aggregator_state[-2] + (1 - self.smoothing) * self.signal_ut

        self.aggregator_state[-2] = self.signal_ut

        # reward = entropy(scaled_action) + self.o1 * charging_performance_reward1 - self.o2
        terminated = False

        if self.timestep >= max_timestep:
            terminated = True
        if not self.train:
            if not terminated:
                self.chosen_ut_for_each_timestep.append(self.signal_ut)
                # self.cumulative_costs += self.costs_list[self.timestep] * self.signal_ut

            self.chosen_sum_of_charging_rates.append(math.fsum(schedule))
        info = {}
        truncated = False

        self.timestep += 1
        self.aggregator_state[-1] += self.delta

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
            # self.chosen_day_index = self.chosen_day_index %
            sampled_day = np.random.choice(list_of_keys,size=1,replace=False)[0]
            self.charging_date = sampled_day
            self.end_charging_date = self.charging_date + time_to_add

            number_of_evs_interval = [30, np.inf]
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
            self.charging_data = evs_timestamp_reset[self.charging_date]

        elif options is not None and ('charging_data' in options.keys()) :
            self.charging_data = options['charging_data']
            # default charging date so we wont get errors
            self.charging_date = datetime(2020,1,1,0,0,0)
            self.end_charging_date = datetime(2020, 1, 1, 23, 59, 59)

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
                    period=self.time_between_timesteps,)
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
        p_t = np.zeros(shape=(self.power_levels,))
        p_t[0] = 1

        self.activated_evse = np.zeros(shape=(self.evse,))
        self.evse_map_to_ev = np.zeros(shape=(self.evse,)) - 1
        # result/debugging vars
        self.chosen_sum_of_charging_rates = []
        self.chosen_ut_for_each_timestep = []
        self.delivered_and_undelivered_energy = {}
        self.dict_arrivals_departures = {}
        self.signal_buffer.clear()
        self.signal_buffer.append(0)
        self.signal_ut = self.ppc(c_t=self.costs_list[self.timestep],p_t=p_t)
        self.aggregator_state[-2] = self.signal_ut
        num_of_timesteps = int((60 * 24) / self.time_between_timesteps)
        self.charging_rates_matrix = np.zeros(shape=(len(self.charging_data),
                                                     num_of_timesteps))

        if not self.train:
            self.chosen_ut_for_each_timestep.append(self.signal_ut)
        info = {}
        return self.aggregator_state, info


# Register the environment
gymnasium.envs.registration.register(
    id='EVenvironment-v0',
    entry_point='environment_ev:EVenvironment',  # Adjust the path as necessary
)


