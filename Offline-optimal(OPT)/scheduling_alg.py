import numpy as np
from datetime import datetime
from preprocessing import get_laxity_of_ev
class SchedulingAlg:
    def __init__(self,
                 EVs:list,
                 start:datetime,
                 end:datetime,
                 available_energy_for_each_timestep,
                 time_between_timesteps=5,
                 accuracy=1e-8,
                 number_of_evse=54,
                 cost_function=None,
                 convert_to_hours=True,
                 process_output=True,
                 costs_loaded_manually:list=None,
                 info_about_future_costs=True,
                 set_time_horizon=True,
                 set_available_energy_for_each_timestep=True,
                 ):
        '''
        Args:
            EVs: all EVs who arrived and departed from charging station withing [start, end]
            start: the time when the charging station opens to customers
            end: the time when charging station closes to customers
            available_energy_for_each_timestep: the energy capacity that charging station has at each timestep of EV charging
            time_between_timesteps: the length of time period between computation of current charging rates and new charging rates
            cost_function: the function that maps charging power of 1kW to costs (based on time)
        '''


        if not isinstance(EVs, list):
            raise ValueError('EVs must be a list.')
        if len(EVs) == 0:
            raise ValueError('EVs list cannot be empty.')
        self.EVs = EVs
        self.EVs_indexed = EVs
        # self.EVs_indexed = [[i] + ev for i, ev in enumerate(self.EVs, start=0)]

        if not isinstance(start, datetime) or not isinstance(end, datetime):
            raise ValueError('Start and end must be datetime objects.')
        if start > end:
            raise ValueError('Start time must sooner than end time.')
        self.start = start
        self.end = end


        if not isinstance(time_between_timesteps, int) and not isinstance(time_between_timesteps, float):
            raise ValueError('Time (in minutes) between consecutive must be float or int.')
        if time_between_timesteps <= 0:
            raise ValueError('Time (in minutes) between consecutive must be above 0.')

        self.time_between_timesteps = time_between_timesteps
        self.charging_timesteps_num = self.get_num_of_charging_timesteps(
            start=start,
            end=end,
            time_between_timesteps=self.time_between_timesteps)

        if not isinstance(number_of_evse, int):
            raise ValueError('Number of EVSE must be an integer.')
        if number_of_evse < 1:
            raise ValueError('Number of EVSE must be an at least 1.')
        self.number_of_evse = number_of_evse
        if set_time_horizon:
            self.time_horizon = [timestep for timestep in range(self.charging_timesteps_num + 1)]
        if not set_time_horizon:
            self.time_horizon = []
        if set_available_energy_for_each_timestep and not isinstance(available_energy_for_each_timestep, list) and not isinstance(available_energy_for_each_timestep, int) and not isinstance(available_energy_for_each_timestep, float):
            raise ValueError('Available energy must be either list, int or float.')

        if set_available_energy_for_each_timestep and isinstance(available_energy_for_each_timestep, list):
            if len(available_energy_for_each_timestep) != len(self.time_horizon):
                raise ValueError('Available energy list must be of same length as time horizon.')
            self.available_energy_for_each_timestep = available_energy_for_each_timestep

        if (set_available_energy_for_each_timestep and
                (isinstance(available_energy_for_each_timestep, int) or isinstance(available_energy_for_each_timestep, float))):
            # if available_energy_for_each_timestep <= 0:
            #     raise ValueError('Available energy must be above 0.')
            self.available_energy_for_each_timestep = np.zeros(
                shape=(len(self.time_horizon))) + available_energy_for_each_timestep
        if not set_available_energy_for_each_timestep:
            self.available_energy_for_each_timestep = []
        if cost_function is not None and costs_loaded_manually is not None:
            raise ValueError('Choose costs generated by cost function or manually computed costs, not both.')

        if cost_function is None and costs_loaded_manually is None:
            raise ValueError('Choose costs generated by cost function or manually computed costs, nothing was chosen.')

        self.cost_function = self.default_cost_function
        self.convert_to_hours = convert_to_hours
        if cost_function is not None:
            self.cost_function = self.cost_function
            self.cost_vector = np.zeros(shape=len(self.time_horizon))
            for i, t in enumerate(self.time_horizon, start=0):
                self.cost_vector[t] = self.cost_function(timestep=t,
                                                         time_between_timesteps= self.time_between_timesteps,
                                                         convert_to_hours=self.convert_to_hours)

        if costs_loaded_manually is not None:
            self.cost_function = None
            if info_about_future_costs and len(costs_loaded_manually) != len(self.time_horizon):
                raise ValueError('Manually computed costs list must be of same length as time horizon')
            self.cost_vector = costs_loaded_manually

        self.charging_plan_for_all_ev = np.zeros((len(self.EVs), len(self.time_horizon)))
        self.process_output = process_output
        self.accuracy = accuracy
        self.info_about_future_costs = info_about_future_costs

    def get_active_evs_connected_to_evses(self,
                                          evs:list,
                                          timestep:int,
                                          number_of_evse=54):
        number_of_taken_evse = 0
        res_evs = []
        sorted_evs = sorted(evs, key=lambda x: x[1])
        for ev in sorted_evs:
            index, ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = ev
            if number_of_taken_evse > number_of_evse:
                raise ValueError('Not enough EVSE for solving the problem, raise number of EVSE')
            if ev_arrival <= timestep <= ev_departure:
                res_evs.append(ev)
                number_of_taken_evse += 1
        return res_evs

    def sort_evs(self, evs,
                 evs_remaining_energy_to_be_charged,
                 current_timestep,
                 sorting_fun=get_laxity_of_ev):
        evs_array = []
        res_array = []
        for ev in evs:
            index, ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = ev
            ev_remaining_energy_to_be_charged = evs_remaining_energy_to_be_charged[index]
            calculated_value = sorting_fun(ev=ev,
                                           ev_remaining_energy_to_be_charged=ev_remaining_energy_to_be_charged,
                                           current_timestep=current_timestep)
            evs_array.append([calculated_value] + ev)
        evs_array = sorted(evs_array, key = lambda x: x[0])
        for ev in evs_array:
            res_array.append(ev[1:])
        return res_array
    def _validate_time_between_timesteps(self, time_between_timesteps):
        if not isinstance(time_between_timesteps, int) and not isinstance(time_between_timesteps, float):
            raise ValueError('Time_between_timesteps must be int or float.')
        if time_between_timesteps <= 0:
            raise ValueError('Time_between_timesteps must be greater than 0.')

    def get_num_of_charging_timesteps(self,
                                      start:datetime,
                                      end:datetime,
                                      time_between_timesteps):
        if start > end:
            raise ValueError('Start must be sooner or at same time as the end.')
        self._validate_time_between_timesteps(time_between_timesteps)
        charging_timesteps_num = (end - start).seconds + ((24*60*60) * ((end - start).days))
        charging_timesteps_num /= 60
        charging_timesteps_num /= time_between_timesteps
        charging_timesteps_num = int(charging_timesteps_num)
        return charging_timesteps_num



    def convert_timestep_to_hours(self, timestep,
                                   time_between_timesteps):
        self._validate_time_between_timesteps(time_between_timesteps)
        return timestep * (time_between_timesteps / 60)

    def default_cost_function(self, timestep,
                              time_between_timesteps=5,
                              convert_to_hours=False,
                              daily_prices=True):
        self._validate_time_between_timesteps(time_between_timesteps)
        timesteps_in_one_day = int((24 * 60) / time_between_timesteps)
        costs = 0
        if daily_prices:
            costs = timestep % timesteps_in_one_day
        if not daily_prices:
            costs = timestep
        if convert_to_hours:
            costs = self.convert_timestep_to_hours(timestep=costs,
                                                   time_between_timesteps=time_between_timesteps)


        return costs

    def inverse_cost_function(self, timestep,
                              time_between_timesteps=5,
                              convert_to_hours = False
                              ):
        self._validate_time_between_timesteps(time_between_timesteps)
        timesteps_in_one_day = int((24 * 60) / time_between_timesteps)
        costs = timestep % timesteps_in_one_day
        if convert_to_hours:
            costs = self.convert_timestep_to_hours(timestep=costs,
                                                   time_between_timesteps=time_between_timesteps)
            costs /= 24
        return 1 - costs

    # different definitions of laxity in articles - what should i use?
    # def laxity_cost_function(self, timestep, ev, ev_remaining_energy_to_be_charged):
    #     return get_laxity_of_ev(
    #         ev=ev,
    #         ev_remaining_energy_to_be_charged=ev_remaining_energy_to_be_charged,
    #         current_timestep=timestep)

    def constant_cost_function(self):
        return 1

    def unfair_cost_function(self, timestep,
                              time_between_timesteps=5,
                              convert_to_hours = False):
        self._validate_time_between_timesteps(time_between_timesteps)
        timesteps_in_one_day = int((24 * 60) / time_between_timesteps)
        costs = timestep % timesteps_in_one_day
        if convert_to_hours:
            costs = self.convert_timestep_to_hours(timestep=costs,
                                                   time_between_timesteps=time_between_timesteps)
        res = 10000 if costs < 13 else 20
        return res



