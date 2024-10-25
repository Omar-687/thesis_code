from typing import Optional
import numpy as np
import gymnasium
from gymnasium import spaces
from tensorflow.python.keras.saving.utils_v1.mode_keys import is_train
from scipy.stats import entropy

class EVenvironment(gymnasium.Env):

    def __init__(self,
                 scheduling_algorithm,
                 charging_days_list,
                 cost_list,
                 evse=54,
                 tuning_parameter=6e3,
                 max_charging_rate=6.6,
                 power_levels=10,
                 time_between_timesteps=12,
                 power_rating=150,
                 o1=0.1,
                 o2=0.2,
                 o3=2):
        self.scheduling_algorithm = scheduling_algorithm
        low_vector = np.zeros(shape=(evse*2+1,))
        high_vector = np.zeros(shape=(evse*2+1,))
        for i in range(evse*2+1):
            if i % 2 == 0:
                high_vector[i] = 24
            elif i % 2 == 1:
                high_vector[i] = 60
        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self.observation_space = spaces.Box(low=low_vector, high=high_vector)
        low_bound_action_space = np.zeros(shape=(power_levels,))
        upper_bound_action_space = np.ones(shape=(power_levels,))
        self.action_space = spaces.Box(low=low_bound_action_space, high=upper_bound_action_space)
        self.charging_days_list = charging_days_list
        self.power_levels = power_levels
        self.costs_list = cost_list
        self.time_between_timesteps = time_between_timesteps
        self.power_rating = power_rating
        self.max_charging_rate = max_charging_rate
        self.tuning_parameter = tuning_parameter
        self.evse = evse
        self.timestep = 0
        self.o1 = 0.1
        self.o2 = 0.2
        self.o3 = 2
        self.chosen_power_levels = []
        self.chosen_charging_rates = []
    def ppc(self, c_t, p_t):
        if np.all(p_t == 0):
            return 0
        low = 0
        high = self.power_rating
        u_ts = np.random.uniform(low=low, high=high, size=self.power_levels)
        min_value = np.zeros(shape=(2,)) + np.inf
        for i in range(self.power_levels):
            value = c_t(u_ts[i]) - self.tuning_parameter * np.log(p_t[i])
            if value > min_value[1]:
                min_value = i, value
        best_ut_index = min_value[0]
        return u_ts[best_ut_index]

    def find_free_evse(self):
        for i in range(0,len(self.aggregator_state) - 1, 2):
            if self.aggregator_state[i] == 0:
                return i
        return None
    # if costs were real time, at each timestep they would be loaded
    def step(self, action):
        self.timestep += 1
        self.aggregator_state[-1] += (self.time_between_timesteps / 60)
        scaled_action = action / np.sum(action)
        charging_decisions_vector = np.zeros(shape=(self.evse,))
        activated_evse = np.zeros(shape=(self.evse,))
        # TODO: where to use PPC?
        for ev in self.charging_data:
            index, arrival, departure, maximum_charging_rate, requested_energy = ev
            if self.timestep == arrival:
                ...

        not_charged_penalty = 0
        for i in self.aggregator_state:
            if i % 2 == 0 and activated_evse[i//2] == 1 and self.aggregator_state[i] == 0:
                ...
            if i % 2 == 0 and activated_evse[i//2] == 1:
                self.aggregator_state[i] -= (self.time_between_timesteps / 60)
            if i % 2 == 1 and activated_evse[i//2] == 1:
                self.aggregator_state[i] -= charging_decisions_vector[i//2]


        charging_performance_reward1 = np.sum(charging_decisions_vector)


        reward = entropy(scaled_action) + self.o1 * charging_performance_reward1 - self.o2
        terminated = False
        if self.aggregator_state >= 24:
            terminated = True

        #
        #
        # reward =
        return self.aggregator_state, reward, terminated
        # return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.aggregator_state = np.zeros(self.evse*2+1)
        sampled_elements_with_replacement = np.random.choice(self.charging_days_list)
        self.charging_date = sampled_elements_with_replacement
        # TODO: access the data via timeseries
        self.charging_data = self.charging_date
        self.timestep = 0
        # TODO: what else should i return
        return self.aggregator_state
