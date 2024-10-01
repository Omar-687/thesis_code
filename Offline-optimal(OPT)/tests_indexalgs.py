from datetime import datetime, timedelta
import math

from adacharge import cost_function
from scheduling_alg import SchedulingAlg
from index_based_algs import *
from utils import *
import networks
import unittest
from testing_functions import (check_all_energy_demands_met,
                               check_infrastructure_not_violated,
                               check_charging_rates_within_bounds,
                               check_number_of_taken_evse)


from os.path import exists
# TODO: test on both datasets (including static ACN)
# TODO: test infeasible solutions
class testLLF(unittest.TestCase):
    def test1(self):
        filename = 'acndata_sessions_acn.json'
        txt_ev_info_filename = 'evs_data.txt'
        settings_filename = 'settings.txt'
        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 26, 23, 59, 59)

        period = 5
        evs, evs_time_not_normalised = get_evs_data_from_document(document=filename,
                                                                  start=start,
                                                                  end=end,
                                                                  period=period,
                                                                  amount_of_evs_interval=[5, 10])
        cost_function = SchedulingAlg.default_cost_function
        save_evs_to_file(filename=txt_ev_info_filename,
                         evs=evs,
                         evs_with_time_not_normalised=evs_time_not_normalised)
        number_of_evse = 54
        available_energy_for_each_timestep = bisection_search_Pt(
            EVs=evs,
            start=start,
            end=end,
            algorithm=LeastLaxityFirstAlg,
            number_of_evse=number_of_evse,
            period=period,
            cost_function=cost_function)

        scheduling_alg = (
            LeastLaxityFirstAlg(
                EVs=evs,
                start=start,
                end=end,
                available_energy_for_each_timestep=available_energy_for_each_timestep,
                time_between_timesteps=period,
                cost_function=cost_function))

        cost_vector = scheduling_alg.cost_vector
        feasibility, charging_rates = scheduling_alg.solve()

        create_settings_file(evs_num=len(evs),
                             filename=settings_filename,
                             start=start,
                             end=end,
                             available_energy_for_each_timestep=available_energy_for_each_timestep,
                             time_horizon=scheduling_alg.time_horizon,
                             period=period,
                             number_of_evse=number_of_evse,
                             charging_network=['caltech'],
                             garages=caltech_garages,
                             algorithm_name=scheduling_alg.algorithm_name)

        charging_rates_within_bounds = check_charging_rates_within_bounds(evs=evs,
                                                                          charging_rates=charging_rates,
                                                                          )
        self.assertTrue(charging_rates_within_bounds)

        infrastructure_not_violated = check_infrastructure_not_violated(charging_rates=charging_rates,
                                                                        available_energy_for_each_timestep=scheduling_alg.available_energy_for_each_timestep)

        self.assertTrue(infrastructure_not_violated)

        all_energy_demands_met = check_all_energy_demands_met(evs=evs,
                                                              charging_rates=charging_rates,
                                                              algorithm_name=scheduling_alg.algorithm_name)
        self.assertTrue(all_energy_demands_met)

        number_of_taken_evse_not_exceeded = check_number_of_taken_evse(charging_rates=charging_rates,
                                                                       number_of_evse=scheduling_alg.number_of_evse)

        self.assertTrue(number_of_taken_evse_not_exceeded)
        create_table(charging_profiles_matrix=charging_rates,
                     charging_cost_vector=cost_vector,
                     period=period,
                     show_charging_costs=True,
                     )


    def test1_infeasible(self):
        filename = 'acndata_sessions_acn.json'
        txt_ev_info_filename = 'evs_data.txt'
        settings_filename = 'settings.txt'
        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 26, 23, 59, 59)

        period = 5
        evs, evs_time_not_normalised = get_evs_data_from_document(document=filename,
                                                                  start=start,
                                                                  end=end,
                                                                  period=period,
                                                                  amount_of_evs_interval=[5, 10])
        cost_function = SchedulingAlg.default_cost_function
        number_of_evse = 1
        save_evs_to_file(filename=txt_ev_info_filename,
                         evs=evs,
                         evs_with_time_not_normalised=evs_time_not_normalised)

        available_energy_for_each_timestep = 10

        scheduling_alg = (
            LeastLaxityFirstAlg(EVs=evs,
                                start=start,
                                end=end,
                                available_energy_for_each_timestep=available_energy_for_each_timestep,
                                time_between_timesteps=period,
                                number_of_evse=number_of_evse,
                                cost_function=cost_function))

        cost_vector = scheduling_alg.cost_vector
        feasibility, charging_rates = scheduling_alg.solve()
        self.assertFalse(feasibility)
        create_settings_file(
            evs_num=len(evs),
            filename=settings_filename,
            start=start,
            end=end,
            period=period,
            available_energy_for_each_timestep=available_energy_for_each_timestep,
            time_horizon=scheduling_alg.time_horizon,
            number_of_evse=number_of_evse,
            charging_network=['caltech'],
            garages=caltech_garages,
            algorithm_name=scheduling_alg.algorithm_name,

        )
        charging_rates_within_bounds = check_charging_rates_within_bounds(evs=evs,
                                                                          charging_rates=charging_rates,
                                                                          )
        self.assertTrue(charging_rates_within_bounds)

        infrastructure_not_violated = check_infrastructure_not_violated(charging_rates=charging_rates,
                                                                        available_energy_for_each_timestep=scheduling_alg.available_energy_for_each_timestep)

        self.assertTrue(infrastructure_not_violated)


    def test2(self):
        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 29, 23, 59, 59)
        period = 5
        filename = 'acndata_sessions_acn.json'
        evs, evs_time_not_normalised = get_evs_data_from_document(document=filename,
                                                                  start=start,
                                                                  end=end,
                                                                  period=period,
                                                                  amount_of_evs_interval=None)
        cost_function = SchedulingAlg.default_cost_function
        available_energy_for_each_timestep = (
            bisection_search_Pt(EVs=evs,
                                start=start,
                                end=end,
                                period=period,
                                algorithm=LeastLaxityFirstAlg,
                                cost_function=cost_function))
        scheduling_alg = LeastLaxityFirstAlg(EVs=evs,
                                                start=start,
                                                end=end,
                                                available_energy_for_each_timestep=available_energy_for_each_timestep,
                                                time_between_timesteps=period,
                                                cost_function=cost_function)
        feasibility, charging_rates = scheduling_alg.solve()
        charging_rates_within_bounds = check_charging_rates_within_bounds(evs=evs,
                                                                          charging_rates=charging_rates,
                                                                          )
        self.assertTrue(charging_rates_within_bounds)

        infrastructure_not_violated = check_infrastructure_not_violated(charging_rates=charging_rates,
                                                                        available_energy_for_each_timestep=scheduling_alg.available_energy_for_each_timestep)

        self.assertTrue(infrastructure_not_violated)

        all_energy_demands_met = check_all_energy_demands_met(evs=evs,
                                                              charging_rates=charging_rates,
                                                              algorithm_name=scheduling_alg.algorithm_name)
        self.assertTrue(all_energy_demands_met)

        number_of_taken_evse_not_exceeded = check_number_of_taken_evse(charging_rates=charging_rates,
                                                                       number_of_evse=scheduling_alg.number_of_evse)

        self.assertTrue(number_of_taken_evse_not_exceeded)