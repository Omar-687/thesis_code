import copy
import math
from datetime import datetime, timedelta
import numpy as np
from preprocessing import get_active_evs, get_laxity_of_ev
from postprocessing import is_solution_feasible, correct_charging_rate
from scheduling_alg import SchedulingAlg
# TODO: add the possibility of setting capacity as an array
# hard to test with other results than from bisection-generated output
class SmoothedLeastLaxityAlg(SchedulingAlg):
    def __init__(self, EVs,
                       start,
                       end,
                       available_energy_for_each_timestep,
                       time_between_timesteps=5,
                       accuracy=1e-8,
                       number_of_evse=54,
                       cost_function=None,
                       process_output=True,
                       costs_loaded_manually=None
                       ):
        super().__init__(EVs=EVs,
                         start=start,
                         end=end,
                         available_energy_for_each_timestep=available_energy_for_each_timestep,
                         time_between_timesteps=time_between_timesteps,
                         accuracy=accuracy,
                         number_of_evse=number_of_evse,
                         cost_function=cost_function,
                         process_output=process_output,
                         costs_loaded_manually=costs_loaded_manually)
        self.algorithm_name = 'sLLF'

    def get_laxities(self, EVs:list, evs_remaining_energy_to_be_charged:dict, current_timestep):
        laxities_of_all_evs = np.zeros(shape=(len(evs_remaining_energy_to_be_charged.keys())))
        for ev in EVs:
            index, arrival, departure, maximum_charging_rate, requested_energy = ev
            ev_remaining_energy_to_be_charged = evs_remaining_energy_to_be_charged[index]
            laxity = get_laxity_of_ev(ev=ev,
                             ev_remaining_energy_to_be_charged=ev_remaining_energy_to_be_charged,
                             current_timestep=current_timestep)
            laxities_of_all_evs[index] = laxity
        return laxities_of_all_evs

    def get_schedule_given_L_t(self,
                               active_EVs,
                               L_t,
                               evs_laxities,
                               evs_remaining_energy_to_be_charged):
        charging_rates = np.zeros(shape=(len(self.EVs)))
        for active_EV in active_EVs:
            index, ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = active_EV
            charging_rates[index] = ev_maximum_charging_rate * (L_t - evs_laxities[index] + 1)
            charging_rates[index] = correct_charging_rate(
                charging_rate=charging_rates[index],
                ev_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged[index],
                maximum_charging_rate=ev_maximum_charging_rate
            )
        return charging_rates

    def optimization_problem_bisection_solution(self,
                                                active_EVs:list,
                                                evs_laxities:list,
                                                evs_remaining_energy_to_be_charged:dict,
                                                timestep
                                                ):
        lower_bound_Lt = min(evs_laxities) - 1
        upper_bound_Lt = max(evs_laxities)
        while abs(upper_bound_Lt - lower_bound_Lt) > self.accuracy:
            middle_Lt = (upper_bound_Lt + lower_bound_Lt) / 2
            charging_rates = self.get_schedule_given_L_t(
                active_EVs=active_EVs,
                L_t=middle_Lt,
                evs_laxities=evs_laxities,
                evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged)

            given_energy = math.fsum(charging_rates)
            if given_energy > self.available_energy_for_each_timestep[timestep]:
                upper_bound_Lt = middle_Lt
            else:
                lower_bound_Lt = middle_Lt
        return lower_bound_Lt

    def solve_for_current_timestep(self, current_timestep:int, evs_remaining_energy_to_be_charged:dict):
        active_EVs = self.get_active_evs_connected_to_evses(evs=self.EVs_indexed,
                                                            evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged,
                                                            timestep=current_timestep,
                                                            number_of_evse=self.number_of_evse)
        if len(active_EVs) == 0:
            return self.charging_plan_for_all_ev, evs_remaining_energy_to_be_charged

        evs_laxities = self.get_laxities(EVs=active_EVs,
                                         evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged,
                                         current_timestep=current_timestep)
        optimal_L_t = self.optimization_problem_bisection_solution(
            active_EVs=active_EVs,
            evs_laxities=evs_laxities,
            evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged,
            timestep=current_timestep)

        self.charging_plan_for_all_ev[:, current_timestep] = self.get_schedule_given_L_t(
                active_EVs=active_EVs,
                L_t=optimal_L_t,
                evs_laxities=evs_laxities,
                evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged)
        # for key, value in evs_remaining_energy_to_be_charged:
        for ev in active_EVs:
            index = ev[0]
            evs_remaining_energy_to_be_charged[index] -= self.charging_plan_for_all_ev[index, current_timestep]
        return self.charging_plan_for_all_ev, evs_remaining_energy_to_be_charged

    def solve(self):
        evs_remaining_energy_to_be_charged = {}
        # charging_plan_for_all_ev = np.zeros((len(self.EVs), len(self.time_horizon)))
        for ev in self.EVs_indexed:
            index, ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = ev
            evs_remaining_energy_to_be_charged[index] = ev_requested_energy


        for current_timestep in range(len(self.time_horizon)):
            # active_EVs = get_active_evs(evs=self.EVs_indexed, current_timestep=current_timestep)
            active_EVs = self.get_active_evs_connected_to_evses(evs=self.EVs_indexed,
                                                                evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged,
                                                                timestep=current_timestep,
                                                                number_of_evse=self.number_of_evse)
            if len(active_EVs) == 0:
                continue
            evs_laxities = self.get_laxities(EVs=active_EVs,
                                             evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged,
                                             current_timestep=current_timestep)

            optimal_L_t = self.optimization_problem_bisection_solution(
                active_EVs=active_EVs,
                evs_laxities=evs_laxities,
                evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged,
                timestep=current_timestep)

            self.charging_plan_for_all_ev[:, current_timestep] = self.get_schedule_given_L_t(
                active_EVs=active_EVs,
                L_t=optimal_L_t,
                evs_laxities=evs_laxities,
                evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged)
            for ev in active_EVs:
                index = ev[0]
                evs_remaining_energy_to_be_charged[index] -=  self.charging_plan_for_all_ev[index, current_timestep]

        feasibility = is_solution_feasible(
            EVs=self.EVs,
            charging_rates=self.charging_plan_for_all_ev,
            available_energy_for_each_timestep=self.available_energy_for_each_timestep,
            algorithm_name=self.algorithm_name)
        return feasibility, self.charging_plan_for_all_ev





