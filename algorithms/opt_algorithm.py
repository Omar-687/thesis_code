import copy

import numpy as np
import cvxpy as cp
# implementation of OPT algorithm from Optimal Online Adaptive Electric Vehicle Charging
from datetime import datetime, timedelta
#
# one day has 86400 seconds
class OPT:
    def __init__(self,
                 EVs,
                 start:datetime,
                 end:datetime,
                 available_energy_for_each_timestep,
                 time_between_timesteps=5,
                 cost_function=None,
                 process_output=True
                 ):
        self.EVs = EVs
        self.available_energy_for_each_timestep = available_energy_for_each_timestep

        self.process_output = process_output
        self.time_between_timesteps = time_between_timesteps
        self.charging_timesteps_num = (end - start).seconds * ((end - start).days + 1)
        self.charging_timesteps_num /= 60
        self.charging_timesteps_num /= self.time_between_timesteps
        self.charging_timesteps_num = int(self.charging_timesteps_num)
        self.time_horizon = [timestep for timestep in range(self.charging_timesteps_num +1)]
        # self.T = [timestep for timestep in range(num_of_days * (60//self.time_between_timesteps) * 24)]
        if cost_function is None:
            self.cost_function = self.default_cost_function
        else:
            self.cost_function = self.cost_function

    def set_cost_function(self, cost_function):
        self.cost_function = cost_function
    # 2) in Impact of cost function
    def default_cost_function(self, time):
        return time
    def get_cost_vector(self):
        # real_time_mins = 0
        shape = (len(self.time_horizon))
        price_vector = np.zeros(shape=shape)



        for i, t in enumerate(self.time_horizon, start=0):
            # real_time_mins += self.time_between_timesteps
            # real_time_hours = (real_time_mins / 60) % 24
            price_vector[i] = self.cost_function(t)
        return price_vector




    def solve(self, verbose=False):
        constraints = []
        # matrix N*T
        all_charging_profiles_rn = cp.Variable(shape=(len(self.EVs), len(self.time_horizon)))
        lb = np.zeros(shape=(len(self.EVs), len(self.time_horizon)))
        all_Pts = np.zeros(shape=(len(self.time_horizon))) + self.available_energy_for_each_timestep

        all_ev_demanded_energy = np.zeros(shape=(len(self.EVs)))
        all_ev_maximum_charging_rate_matrix = np.zeros(shape=(len(self.EVs), len(self.time_horizon)))
        for i, ev in enumerate(self.EVs, start=0):
            ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = ev
            all_ev_demanded_energy[i] = ev_requested_energy
            all_ev_maximum_charging_rate_matrix[i, ev_arrival : ev_departure + 1] += ev_maximum_charging_rate

        constraints.append(lb <= all_charging_profiles_rn)
        constraints.append(all_charging_profiles_rn <= all_ev_maximum_charging_rate_matrix)
        constraints.append(cp.sum(all_charging_profiles_rn, axis=1) == all_ev_demanded_energy)
        constraints.append(cp.sum(all_charging_profiles_rn, axis=0) <= all_Pts)

        price_vector = self.get_cost_vector()
        price_matrix = price_vector.reshape((price_vector.shape[0], 1))
        objective = all_charging_profiles_rn @ price_vector
        objective = cp.sum(cp.sum(objective, axis=0))
        prob = cp.Problem(cp.Minimize(objective), constraints)
        # default solver without specifying is ECOS - a LP solver
        # prob.solve()
        prob.solve(solver=cp.ECOS, verbose=verbose)

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
                for i in range(len(all_charging_profiles_rn.value)):
                    ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = self.EVs[i]
                    for j in range(len(all_charging_profiles_rn.value[i])):
                        if all_charging_profiles_rn.value[i][j] < 0 or all_ev_maximum_charging_rate_matrix[i][j] == 0:
                            res[i][j] = 0
                        elif all_charging_profiles_rn.value[i][j] > all_ev_maximum_charging_rate_matrix[i][j]:
                            res[i][j] = all_ev_maximum_charging_rate_matrix[i][j]
                        else:
                            res[i][j] = all_charging_profiles_rn.value[i][j]

                        if res[i][j] > ev_requested_energy:
                            res[i][j] = ev_requested_energy
                        ev_requested_energy -= res[i][j]

            else:
                res = copy.deepcopy(all_charging_profiles_rn)
        else:
            return [False, None]



        return [True, res]






