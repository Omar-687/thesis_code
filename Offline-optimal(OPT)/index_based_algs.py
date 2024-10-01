import copy
from preprocessing import get_active_evs, get_laxity_of_ev
from postprocessing import is_solution_feasible
import numpy as np
from scheduling_alg import SchedulingAlg

class LeastLaxityFirstAlg(SchedulingAlg):
    def __init__(self,
                 EVs,
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
        self.algorithm_name = 'LLF'


    # we calculate this by using bisection
    # optimisation variable is charging rate
    # validate if this is ok once again with other articles about bisection
    def find_maximum_feasible_charging_rate(self, ev,
                                            remaining_available_energy_at_given_timestep,
                                            ev_remaining_energy_to_be_charged,
                                            timestep
                                            ):
        index, ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = ev
        minimum_value = 0
        maximum_value = min(ev_maximum_charging_rate,ev_remaining_energy_to_be_charged)
        while abs(maximum_value - minimum_value) > self.accuracy:
            maximum_feasible_charging_rate_of_ev = (maximum_value + minimum_value) / 2
            if maximum_feasible_charging_rate_of_ev > remaining_available_energy_at_given_timestep[timestep]:
                maximum_value = maximum_feasible_charging_rate_of_ev
            else:
                minimum_value = maximum_feasible_charging_rate_of_ev

        return minimum_value


    def solve_for_one_timestep(
            self,
            current_timestep,
            evs_remaining_energy_to_be_charged):

        active_evs = self.get_active_evs_connected_to_evses(
            evs=self.EVs_indexed,
            evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged,
            timestep=current_timestep,
            number_of_evse=self.number_of_evse)

        if len(active_evs) == 0:
            return self.charging_plan_for_all_ev, evs_remaining_energy_to_be_charged


        sorted_active_evs = self.sort_evs(
            evs=active_evs,
            evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged,
            current_timestep=current_timestep)

        remaining_available_energy_at_given_timestep = copy.deepcopy(self.available_energy_for_each_timestep)
        for ev in sorted_active_evs:
            index = ev[0]
            ev_remaining_energy_to_be_charged = evs_remaining_energy_to_be_charged[index][1]
            maximum_feasible_charging_rate = self.find_maximum_feasible_charging_rate(
                ev=ev,
                remaining_available_energy_at_given_timestep=remaining_available_energy_at_given_timestep,
                ev_remaining_energy_to_be_charged=ev_remaining_energy_to_be_charged,
                timestep=current_timestep)
            remaining_available_energy_at_given_timestep[current_timestep] -= maximum_feasible_charging_rate
            evs_remaining_energy_to_be_charged[index] -= maximum_feasible_charging_rate
            self.charging_plan_for_all_ev[index, current_timestep] = maximum_feasible_charging_rate

        return self.charging_plan_for_all_ev


    def solve(self):
        evs_remaining_energy_to_be_charged = {}
        # charging_plan_for_all_ev = np.zeros((len(self.EVs), len(self.time_horizon)))
        for ev in self.EVs_indexed:
            index, ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = ev
            evs_remaining_energy_to_be_charged[index] = ev_requested_energy
            # evs_remaining_energy_to_be_charged.append([index, ev_requested_energy])

        for current_timestep in self.time_horizon:
            active_evs = self.get_active_evs_connected_to_evses(
                evs=self.EVs_indexed,
                evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged,
                timestep=current_timestep,
                number_of_evse=self.number_of_evse)

            if len(active_evs) == 0:
                continue

            sorted_active_evs = self.sort_evs(evs=active_evs,
                          evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged,
                          current_timestep=current_timestep)
            remaining_available_energy_at_given_timestep = copy.deepcopy(self.available_energy_for_each_timestep)
            for ev in sorted_active_evs:
                index = ev[0]
                ev_remaining_energy_to_be_charged = evs_remaining_energy_to_be_charged[index]
                maximum_feasible_charging_rate = self.find_maximum_feasible_charging_rate(
                    ev=ev,
                    remaining_available_energy_at_given_timestep=remaining_available_energy_at_given_timestep,
                    ev_remaining_energy_to_be_charged=ev_remaining_energy_to_be_charged,
                    timestep=current_timestep)
                remaining_available_energy_at_given_timestep[current_timestep] -= maximum_feasible_charging_rate
                evs_remaining_energy_to_be_charged[index] -= maximum_feasible_charging_rate
                self.charging_plan_for_all_ev[index, current_timestep] = maximum_feasible_charging_rate


        feasibility = is_solution_feasible(EVs=self.EVs,
                                           charging_rates=self.charging_plan_for_all_ev,
                                           available_energy_for_each_timestep=self.available_energy_for_each_timestep,
                                           algorithm_name=self.algorithm_name)
        return feasibility, self.charging_plan_for_all_ev


