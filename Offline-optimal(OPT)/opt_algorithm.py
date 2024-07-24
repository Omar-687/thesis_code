import numpy as np
import cvxpy as cp
# implementation of OPT algorithm from Optimal Online Adaptive Electric Vehicle Charging
#
class OPT:
    def __init__(self,
                 EVs,
                 # kW capacity
                 P_t,
                 # time between timesteps is in minutes
                 time_between_timesteps=5,
                 num_of_days=1,
                 price_function=None
                 ):
        self.EVs = EVs
        self.P = P_t
        self.length_of_charging = num_of_days
        self.time_between_timesteps = time_between_timesteps
        self.T = [timestep for timestep in range(num_of_days * (60//self.time_between_timesteps) * 24)]
        if price_function is None:
            self.price_function = self.default_price_function
        else:
            self.price_function = self.price_function

    def set_price_function(self, price_function):
        self.price_function = price_function
    # 2) in Impact of cost function
    def default_price_function(self, time):
        return time
    def get_price_vector(self):
        # real_time_mins = 0
        shape = (len(self.T))
        price_vector = np.zeros(shape=shape)



        for i, t in enumerate(self.T, start=0):
            # real_time_mins += self.time_between_timesteps
            # real_time_hours = (real_time_mins / 60) % 24
            price_vector[i] = self.price_function(t)
        return price_vector




    def solve(self):
        constraints = []
        # matrix N*T
        all_charging_profiles_rn = cp.Variable(shape=(len(self.EVs), len(self.T)))
        lb = np.zeros(shape=(len(self.EVs), len(self.T)))
        all_Pts = np.zeros(shape=(len(self.T))) + self.P

        all_ev_demanded_energy = np.zeros(shape=(len(self.EVs)))
        all_ev_peak_charging_rate = np.zeros(shape=(len(self.EVs), len(self.T)))
        for i, ev in enumerate(self.EVs, start=0):
            ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = ev
            all_ev_demanded_energy[i] = ev_requested_energy
            all_ev_peak_charging_rate[i, ev_arrival : ev_departure + 1] += ev_maximum_charging_rate

        constraints.append(lb <= all_charging_profiles_rn)
        constraints.append(all_charging_profiles_rn <= all_ev_peak_charging_rate)
        constraints.append(cp.sum(all_charging_profiles_rn, axis=1) == all_ev_demanded_energy)
        constraints.append(cp.sum(all_charging_profiles_rn, axis=0) <= all_Pts)

        price_vector = self.get_price_vector()
        price_matrix = price_vector.reshape((price_vector.shape[0], 1))
        objective = all_charging_profiles_rn @ price_vector
        objective = cp.sum(cp.sum(objective, axis=0))
        prob = cp.Problem(cp.Minimize(objective), constraints)
        # default solver without specifying is ECOS - a LP solver
        # prob.solve()
        prob.solve(solver=cp.ECOS)

        res = np.zeros(shape=all_charging_profiles_rn.shape)
        if prob.status not in ['infeasible', 'unbounded', 'infeasible_inaccurate']:
        # find out violation tolerance for the solver and what solver is used
        # checking if conditions hold via dual solution(positive entries in dual solution indicate that constraint holds)
        #     for i in range(len(constraints)):
        #         print("A dual solution is")
        #         print(prob.constraints[i].dual_value)
        #  error tolerance 1e-8

            print(prob.status, all_charging_profiles_rn is not None,all_charging_profiles_rn.value is not None)
            for i in range(len(all_charging_profiles_rn.value)):
                for j in range(len(all_charging_profiles_rn.value[i])):

                    if all_charging_profiles_rn.value[i][j] < 0 or all_ev_peak_charging_rate[i][j] == 0:
                        res[i][j] = 0
                    else:
                        res[i][j] = all_charging_profiles_rn.value[i][j]
        else:
            return None



        return res






