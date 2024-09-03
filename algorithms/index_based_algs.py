import copy
from preprocessing import get_active_evs, get_laxity_of_ev
from _pytest.logging import catching_logs
from postprocessing import is_solution_feasible
import numpy as np



class IndexBasedAlg:
    def __init__(self,
                 EVs,
                 start,
                 end,
                 available_energy_for_each_timestep,
                 time_between_timesteps=5,
                 cost_function=None,
                 process_output=True

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
        self.EVs = EVs

        self.EVs_indexed = [[i] + ev for i, ev in enumerate(self.EVs, start=0)]
        self.available_energy_for_each_timestep = available_energy_for_each_timestep
        self.time_between_timesteps = time_between_timesteps
        self.charging_timesteps_num = (end - start).seconds * ((end - start).days + 1)
        self.charging_timesteps_num /= 60
        self.charging_timesteps_num /= self.time_between_timesteps
        self.charging_timesteps_num = int(self.charging_timesteps_num)

        self.time_horizon = [timestep for timestep in range(self.charging_timesteps_num +1)]
        if cost_function is None:
            self.cost_function = self.default_cost_function
        else:
            self.cost_function = self.cost_function
        self.charging_plan_for_all_ev = np.zeros((len(self.EVs), len(self.time_horizon)))

    def set_cost_function(self, cost_function):
        self.cost_function = cost_function
        
    # 2) in Impact of cost function

    def default_cost_function(self, time):
        return time

class LeastLaxityFirstAlg(IndexBasedAlg):
    def __init__(self,
                 EVs,
                 start,
                 end,
                 available_energy_for_each_timestep,
                 time_between_timesteps=5,
                 cost_function=None,
                 process_output=True
                 ):
        super().__init__(EVs=EVs,
                         start=start,
                         end=end,
                         available_energy_for_each_timestep=available_energy_for_each_timestep,
                         time_between_timesteps=time_between_timesteps,
                         cost_function=cost_function,
                         process_output=process_output)

    def get_alg_name(self, short=False):
        return 'LLF' if short else 'least-laxity-first'


    def sort_evs(self, evs,
                 evs_remaining_energy_to_be_charged,
                 current_timestep,
                 sorting_fun=get_laxity_of_ev):
        evs_array = []
        res_array = []
        for ev in evs:
            index, ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = ev
            ev_remaining_energy_to_be_charged = evs_remaining_energy_to_be_charged[index][1]
            calculated_value = sorting_fun(ev=ev,
                                           ev_remaining_energy_to_be_charged=ev_remaining_energy_to_be_charged,
                                           current_timestep=current_timestep)
            evs_array.append([calculated_value] + ev)
        evs_array = sorted(evs_array, key = lambda x: x[0])
        for ev in evs_array:     
            res_array.append(ev[1:]) 
        return res_array

    # we calculate this by using bisection
    # optimisation variable is charging rate
    # validate if this is ok once again with other articles about bisection
    def find_maximum_feasible_charging_rate(self, ev,
                                            remaining_available_energy_at_given_timestep,
                                            ev_remaining_energy_to_be_charged,
                                            # 10e-4 is a bit problematic if we can check the amount of charged energy
                                            error=10e-8
                                            ):
        index, ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = ev
        minimum_value = 0
        maximum_value = ev_maximum_charging_rate
        maximum_feasible_charging_rate_of_ev = (maximum_value + minimum_value) / 2
        while abs(maximum_value - minimum_value) > error:
            maximum_feasible_charging_rate_of_ev = (maximum_value + minimum_value) / 2
            if maximum_feasible_charging_rate_of_ev > remaining_available_energy_at_given_timestep:
                maximum_value = maximum_feasible_charging_rate_of_ev
            elif maximum_feasible_charging_rate_of_ev > ev_remaining_energy_to_be_charged:
                maximum_value = maximum_feasible_charging_rate_of_ev
            else:
                minimum_value = maximum_feasible_charging_rate_of_ev


        # the last variable can be still inaccurate/wrong thats why we take minimum
        return minimum_value




    def solve_for_one_timestep(self, current_timestep, evs_remaining_energy_to_be_charged=None):
        active_evs = get_active_evs(evs=self.EVs_indexed, current_timestep=current_timestep)
        if len(active_evs) == 0:
            return self.charging_plan_for_all_ev, evs_remaining_energy_to_be_charged_curr
        sorted_active_evs = self.sort_evs(evs=active_evs,
                                          evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged,
                                          current_timestep=current_timestep)
        remaining_available_energy_at_given_timestep = self.available_energy_for_each_timestep
        for ev in sorted_active_evs:
            index = ev[0]
            ev_remaining_energy_to_be_charged = evs_remaining_energy_to_be_charged[index][1]
            maximum_feasible_charging_rate = self.find_maximum_feasible_charging_rate(ev=ev,
                                                                                      remaining_available_energy_at_given_timestep=remaining_available_energy_at_given_timestep,
                                                                                      ev_remaining_energy_to_be_charged=ev_remaining_energy_to_be_charged)
            remaining_available_energy_at_given_timestep -= maximum_feasible_charging_rate
            evs_remaining_energy_to_be_charged[index][1] -= maximum_feasible_charging_rate
            self.charging_plan_for_all_ev[index, current_timestep] = maximum_feasible_charging_rate


        return self.charging_plan_for_all_ev


    def solve(self):
        evs_remaining_energy_to_be_charged = []
        # charging_plan_for_all_ev = np.zeros((len(self.EVs), len(self.time_horizon)))
        for ev in self.EVs_indexed:
            index, ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = ev
            evs_remaining_energy_to_be_charged.append([index, ev_requested_energy])

        for current_timestep in self.time_horizon:
            active_evs = get_active_evs(evs=self.EVs_indexed, current_timestep=current_timestep)

            if len(active_evs) == 0:
                continue
            sorted_active_evs = self.sort_evs(evs=active_evs,
                          evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged,
                          current_timestep=current_timestep)
            remaining_available_energy_at_given_timestep = self.available_energy_for_each_timestep
            for ev in sorted_active_evs:
                index = ev[0]
                ev_remaining_energy_to_be_charged = evs_remaining_energy_to_be_charged[index][1]
                maximum_feasible_charging_rate = self.find_maximum_feasible_charging_rate(ev=ev,
                                                          remaining_available_energy_at_given_timestep=remaining_available_energy_at_given_timestep,
                                                          ev_remaining_energy_to_be_charged=ev_remaining_energy_to_be_charged)
                remaining_available_energy_at_given_timestep -= maximum_feasible_charging_rate
                evs_remaining_energy_to_be_charged[index][1] -= maximum_feasible_charging_rate
                self.charging_plan_for_all_ev[index, current_timestep] = maximum_feasible_charging_rate


        feasibility = is_solution_feasible(EVs=self.EVs, charging_rates=self.charging_plan_for_all_ev)
        return [feasibility, self.charging_plan_for_all_ev]


