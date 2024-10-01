import copy

import numpy as np
import cvxpy as cp
# implementation of OPT algorithm from Optimal Online Adaptive Electric Vehicle Charging
from datetime import datetime, timedelta
from scheduling_alg import SchedulingAlg
from numpy.f2py.auxfuncs import throw_error

from postprocessing import (
    correct_charging_rates_offline,
    is_solution_feasible,
    get_maximum_possible_charging_values_given_schedule)
#
# one day has 86400 seconds
class OPT(SchedulingAlg):
    def __init__(self,
                 EVs,
                 start:datetime,
                 end:datetime,
                 available_energy_for_each_timestep,
                 time_between_timesteps=5,
                 accuracy=1e-8,
                 number_of_evse=54,
                 cost_function=None,
                 process_output=True,
                 costs_loaded_manually=None
                 ):
        if accuracy != 1e-8:
            raise ValueError('Accuracy of ECOS solver is 1e-8')
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
        self.algorithm_name = 'OPT'
    # TODO: fix it by not repeatedly adding constraints but progressively increase size of one constraint
    
    def solve(self, verbose=False):
        constraints = []
        # matrix N*T
        all_charging_profiles_rn = cp.Variable(shape=(len(self.EVs), len(self.time_horizon)))
        lb = np.zeros(shape=(len(self.EVs), len(self.time_horizon)))
        # all_Pts = np.zeros(shape=(len(self.time_horizon))) + self.available_energy_for_each_timestep
        all_ev_demanded_energy = np.zeros(shape=(len(self.EVs)))
        all_ev_maximum_charging_rate_matrix = np.zeros(shape=(len(self.EVs), len(self.time_horizon)))
        evs_remaining_energy_to_be_charged = {}
        for ev in self.EVs_indexed:
            index, ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = ev
            all_ev_demanded_energy[index] = ev_requested_energy
            evs_remaining_energy_to_be_charged[index] = ev_requested_energy
            all_ev_maximum_charging_rate_matrix[index, ev_arrival : ev_departure + 1] += ev_maximum_charging_rate

        constraints.append(lb <= all_charging_profiles_rn)
        constraints.append(all_charging_profiles_rn <= all_ev_maximum_charging_rate_matrix)
        constraints.append(cp.sum(all_charging_profiles_rn, axis=1) == all_ev_demanded_energy)
        constraints.append(cp.sum(all_charging_profiles_rn, axis=0) <= self.available_energy_for_each_timestep)
        # add evse if possible
        # price_vector = self.get_cost_vector()
        price_matrix = self.cost_vector.reshape((self.cost_vector.shape[0], 1))
        objective = all_charging_profiles_rn @ self.cost_vector
        objective = cp.sum(cp.sum(objective, axis=0))
        for timestep in self.time_horizon:
            active_evs = self.get_active_evs_connected_to_evses(
                evs=self.EVs_indexed,
                evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged,
                timestep=timestep)
            prob = cp.Problem(cp.Minimize(objective), constraints)

            # default solver without specifying is ECOS - a LP solver
            prob.solve(solver=cp.ECOS, verbose=verbose)
            lb_col = np.zeros(shape=all_charging_profiles_rn[timestep,:].shape)
            ub_col = get_maximum_possible_charging_values_given_schedule(
                active_evs=active_evs,
                schedule=all_charging_profiles_rn[timestep,:],
                maximum_charging_rates_matrix=all_ev_maximum_charging_rate_matrix)
            constraints.append(all_charging_profiles_rn.value[timestep, :])
            if prob.status in ['infeasible', 'unbounded', 'infeasible_inaccurate', 'optimal_inaccurate']:
                return False, None



        res = np.zeros(shape=all_charging_profiles_rn.shape)
        if prob.status not in ['infeasible', 'unbounded', 'infeasible_inaccurate', 'optimal_inaccurate']:
        # find out violation tolerance for the solver and what solver is used
        # checking if conditions hold via dual solution(positive entries in dual solution indicate that constraint holds)
        #     for i in range(len(constraints)):
        #         print("A dual solution is")
        #         print(prob.constraints[i].dual_value)
        #  error tolerance 1e-8

            # print(prob.status, all_charging_profiles_rn is not None,all_charging_profiles_rn.value is not None)

            if self.process_output:
                # move postprocessing
                res = correct_charging_rates_offline(
                    EVs=self.EVs,
                    charging_rates=all_charging_profiles_rn.value,
                    maximum_charging_rates_matrix=all_ev_maximum_charging_rate_matrix
                )

            else:
                res = copy.deepcopy(all_charging_profiles_rn)

            if not is_solution_feasible(
                    EVs=self.EVs,
                    charging_rates=res,
                    available_energy_for_each_timestep=self.available_energy_for_each_timestep,
                    algorithm_name=self.algorithm_name
            ):
                return False, None

            return True, res

        return False, None







