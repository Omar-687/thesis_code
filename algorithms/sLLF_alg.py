import copy
import math
from datetime import datetime, timedelta
import numpy as np
from preprocessing import get_active_evs, get_laxity_of_ev
from postprocessing import is_solution_feasible
class SmoothedLeastLaxityAlg:
    def __init__(self, EVs,
                       start,
                       end,
                       available_energy_for_each_timestep,
                       time_between_timesteps=5,
                       cost_function=None,
                       process_output=True
                       ):
        self.EVs = EVs
        self.EVs_indexed = []
        for i, ev in enumerate(self.EVs, start=0):
            self.EVs_indexed.append([i] + ev)

        self.start = start
        self.end = end
        self.available_energy_for_each_timestep = available_energy_for_each_timestep
        self.time_between_timesteps = time_between_timesteps
        self.cost_function = cost_function

        self.charging_timesteps_num =  (end - start).seconds * ((end - start).days + 1)
        self.charging_timesteps_num /= 60
        self.charging_timesteps_num /= self.time_between_timesteps
        self.charging_timesteps_num = int(self.charging_timesteps_num)
        # self.charging_timesteps_num = int(( ((end - start).seconds ) / 60) / self.time_between_timesteps)

        self.time_horizon = [timestep for timestep in range(self.charging_timesteps_num + 1)]
        self.charging_plan_for_all_ev =  np.zeros((len(self.EVs), len(self.time_horizon)))

    def get_laxities(self, EVs, evs_remaining_energy_to_be_charged, current_timestep):
        laxities_of_all_evs = np.zeros(shape=(len(evs_remaining_energy_to_be_charged)))
        for ev in EVs:
            index, arrival, departure, maximum_charging_rate, requested_energy = ev
            ev_remaining_energy_to_be_charged = evs_remaining_energy_to_be_charged[index]
            laxity = get_laxity_of_ev(ev=ev,
                             ev_remaining_energy_to_be_charged=ev_remaining_energy_to_be_charged,
                             current_timestep=current_timestep)
            laxities_of_all_evs[index] = laxity
        return laxities_of_all_evs

    def optimization_problem_bisection_solution(self,
                                                active_EVs,
                                                evs_laxities,
                                                evs_remaining_energy_to_be_charged,

                                                # big tolerance error_tolerance=10e-4
                                                error_tolerance=10e-8
                                                ):
        charging_rates = np.zeros(shape=(len(self.EVs)))
        lb_Lt = min(evs_laxities) - 1
        up_Lt = max(evs_laxities)
        while abs(up_Lt - lb_Lt) > error_tolerance:
            middle_Lt = (up_Lt + lb_Lt) / 2
            for active_EV in active_EVs:
                index, ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = active_EV
                charging_rates[index] = ev_maximum_charging_rate * (middle_Lt - evs_laxities[index] + 1)

                if charging_rates[index] < 0:
                    charging_rates[index] = 0

                elif charging_rates[index] > min(ev_maximum_charging_rate,
                                             evs_remaining_energy_to_be_charged[index]):
                    charging_rates[index] = min(ev_maximum_charging_rate,
                                            evs_remaining_energy_to_be_charged[index])

            given_energy = math.fsum(charging_rates)
            if given_energy > self.available_energy_for_each_timestep:
                up_Lt = middle_Lt

            else:
                lb_Lt = middle_Lt
        for active_EV in active_EVs:
            index, ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = active_EV
            charging_rates[index] = ev_maximum_charging_rate * (lb_Lt - evs_laxities[index] + 1)

            if charging_rates[index] < 0:
                charging_rates[index] = 0

            elif charging_rates[index] > min(ev_maximum_charging_rate,
                                             evs_remaining_energy_to_be_charged[index]):
                charging_rates[index] = min(ev_maximum_charging_rate,
                                            evs_remaining_energy_to_be_charged[index])

        return charging_rates



    def solve_for_current_timestep(self, current_timestep, evs_remaining_energy_to_be_charged):
        active_EVs = get_active_evs(evs=self.EVs_indexed, current_timestep=current_timestep)
        if len(active_EVs) == 0:
            return self.charging_plan_for_all_ev, evs_remaining_energy_to_be_charged
        evs_laxities = self.get_laxities(EVs=active_EVs,
                                         evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged,
                                         current_timestep=current_timestep)
        current_charging_rates = (
            self.optimization_problem_bisection_solution(active_EVs=active_EVs,
                                                         evs_laxities=evs_laxities,
                                                         evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged,
                                                         error_tolerance=10e-4))
        self.charging_plan_for_all_ev[:, current_timestep] = copy.deepcopy(current_charging_rates)

        new_evs_remaining_energy_to_be_charged = evs_remaining_energy_to_be_charged - self.charging_plan_for_all_ev[:, current_timestep]
        return self.charging_plan_for_all_ev, new_evs_remaining_energy_to_be_charged

    def solve(self):
        evs_remaining_energy_to_be_charged = []
        # charging_plan_for_all_ev = np.zeros((len(self.EVs), len(self.time_horizon)))
        for ev in self.EVs:
            ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = ev
            evs_remaining_energy_to_be_charged.append(ev_requested_energy)


        for current_timestep in range(len(self.time_horizon)):
            active_EVs = get_active_evs(evs=self.EVs_indexed, current_timestep=current_timestep)
            if len(active_EVs) == 0:
                continue
            evs_laxities = self.get_laxities(EVs=active_EVs,
                                             evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged,
                                             current_timestep=current_timestep)
            current_charging_rates = (
                self.optimization_problem_bisection_solution(active_EVs=active_EVs,
                                                             evs_laxities=evs_laxities,
                                                             evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged,
                                                             error_tolerance=10e-4))
            evs_remaining_energy_to_be_charged -= current_charging_rates
            self.charging_plan_for_all_ev[:, current_timestep] = copy.deepcopy(current_charging_rates)

        feasibility = is_solution_feasible(EVs=self.EVs,
                             charging_rates=self.charging_plan_for_all_ev)
        return feasibility, self.charging_plan_for_all_ev






