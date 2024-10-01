from setuptools.command.alias import alias
from acnportal import acnsim, algorithms
import pytz
from datetime import datetime, timedelta
import random
import pandas as pd
import json
import math
from sLLF_alg import SmoothedLeastLaxityAlg
from index_based_algs import *
from utils import *
# import ACNDataStatic
import unittest
from os.path import exists
from testing_functions import (check_all_energy_demands_met,
                               check_charging_rates_within_bounds,
                               check_infrastructure_not_violated)
class testsLLF(unittest.TestCase):
    def test1(self):
        filename = 'acndata_sessions_acn.json'
        txt_ev_info_filename = 'evs_data.txt'
        settings_filename = 'settings.txt'
        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 26, 23, 59, 59)

        period = 5
        number_of_evse = 54
        evs, evs_time_not_normalised = get_evs_data_from_document(document=filename,
                                                                  start=start,
                                                                  end=end,
                                                                  period=period,
                                                                  amount_of_evs_interval=[5, 10])

        save_evs_to_file(filename=txt_ev_info_filename,
                         evs=evs,
                         evs_with_time_not_normalised=evs_time_not_normalised)
        cost_function = SchedulingAlg.default_cost_function
        available_energy_for_each_timestep = bisection_search_Pt(
            EVs=evs,
            start=start,
            end=end,
            period=period,
            algorithm=SmoothedLeastLaxityAlg,
            cost_function=cost_function)

        scheduling_alg = SmoothedLeastLaxityAlg(
            EVs=evs,
            available_energy_for_each_timestep=available_energy_for_each_timestep,
            start=start,
            end=end,
            time_between_timesteps=period,
            cost_function=cost_function)
        cost_vector = scheduling_alg.cost_vector



        feasibility, charging_rates = scheduling_alg.solve()
        check_charging_rates_within_bounds(evs=evs,
                                           charging_rates=charging_rates)


        check_all_energy_demands_met(evs=evs,
                                     charging_rates=charging_rates,
                                     algorithm_name=scheduling_alg.algorithm_name)

        check_infrastructure_not_violated(charging_rates=charging_rates,
                                          available_energy_for_each_timestep=scheduling_alg.available_energy_for_each_timestep)


        create_settings_file(evs_num=len(evs),
                             filename=settings_filename,
                             start=start,
                             end=end,
                             available_energy_for_each_timestep=available_energy_for_each_timestep,
                             time_horizon=scheduling_alg.time_horizon,
                             period=period,
                             algorithm_name='sLLF',
                             charging_network=['caltech'],
                             garages=caltech_garages,
                             number_of_evse=number_of_evse)

        create_table(charging_profiles_matrix=charging_rates,
                     charging_cost_vector=cost_vector,
                     period=period,
                     show_charging_costs=True)
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
                                algorithm=SmoothedLeastLaxityAlg,
                                cost_function=cost_function))
        scheduling_alg = SmoothedLeastLaxityAlg(EVs=evs,
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


    def test_performance_with_cost_functions(self):

        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 29, 23, 59, 59)
        period = 5
        filename = 'acndata_sessions_acn.json'
        evs, evs_time_not_normalised = get_evs_data_from_document(document=filename,
                                                                  start=start,
                                                                  end=end,
                                                                  period=period,
                                                                  amount_of_evs_interval=None)
        cost_functions = [SchedulingAlg.default_cost_function,
                          SchedulingAlg.constant_cost_function,
                          SchedulingAlg.inverse_cost_function]
        # cost_function = SchedulingAlg.default_cost_function
        for cost_function in cost_functions:
            available_energy_for_each_timestep = (
                bisection_search_Pt(EVs=evs,
                                    start=start,
                                    end=end,
                                    period=period,
                                    algorithm=SmoothedLeastLaxityAlg,
                                    cost_function=cost_function))
            scheduling_alg = SmoothedLeastLaxityAlg(EVs=evs,
                                 start=start,
                                 end=end,
                                 available_energy_for_each_timestep=available_energy_for_each_timestep,
                                 time_between_timesteps=period,
                                 cost_function=cost_function)
            feasibility, charging_rates = scheduling_alg.solve()
            charging_rates_within_bounds = check_charging_rates_within_bounds(
                evs=evs,
                charging_rates=charging_rates,
                                                                              )
            self.assertTrue(charging_rates_within_bounds)

            infrastructure_not_violated = check_infrastructure_not_violated(
                charging_rates=charging_rates,
                available_energy_for_each_timestep=scheduling_alg.available_energy_for_each_timestep)

            self.assertTrue(infrastructure_not_violated)

            all_energy_demands_met = check_all_energy_demands_met(
                evs=evs,
                charging_rates=charging_rates,
                algorithm_name=scheduling_alg.algorithm_name)
            self.assertTrue(all_energy_demands_met)

    # def test3(self):
    #     start = datetime(2018, 4, 26, 0, 0, 0)
    #     end = datetime(2018, 4, 26, 23, 59, 59)
    #
    #     # start = datetime(2018, 4, 26, 0, 0, 0)
    #     # end = datetime(2018, 4, 27, 23, 59, 59)
    #     period = 5
    #     # num_of_days = (end - start).days + 1
    #     filename = 'acndata_sessions_acn.json'
    #     evs, evs_time_not_normalised = get_evs_data_from_document(document=filename,
    #                                                               start=start,
    #                                                               end=end,
    #                                                               period=period,
    #                                                               amount_of_evs_interval=None)
    #
    #     available_energy_for_each_timestep = (
    #         bisection_search_Pt(EVs=evs,
    #                             start=start,
    #                             end=end,
    #                             period=period,
    #                             algorithm=SmoothedLeastLaxityAlg))
    #     scheduling_alg = SmoothedLeastLaxityAlg(EVs=evs,
    #                                             start=start,
    #                                             end=end,
    #                                             available_energy_for_each_timestep=available_energy_for_each_timestep,
    #                                             time_between_timesteps=period,
    #                                             cost_function=None)
    #     time_horizon = scheduling_alg.time_horizon
    #
    #     evs_remaining_energy_to_be_charged = []
    #     for ev in scheduling_alg.EVs:
    #         ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = ev
    #         evs_remaining_energy_to_be_charged.append(ev_requested_energy)
    #
    #
    #     charging_rates = scheduling_alg.charging_plan_for_all_ev
    #     for current_timestep in time_horizon:
    #         charging_rates, evs_remaining_energy_to_be_charged = (
    #             scheduling_alg.solve_for_current_timestep(current_timestep=current_timestep,
    #                                                       evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged))
    #
    #     charging_rates_within_bounds = check_charging_rates_within_bounds(evs=evs,
    #                                                                       charging_rates=charging_rates,
    #                                                                       )
    #     self.assertTrue(charging_rates_within_bounds)
    #
    #     infrastructure_not_violated = check_infrastructure_not_violated(charging_rates=charging_rates,
    #                                                                     available_energy_for_each_timestep=available_energy_for_each_timestep)
    #
    #     self.assertTrue(infrastructure_not_violated)
    #
    #     all_energy_demands_met = check_all_energy_demands_met(evs=evs,
    #                                                           charging_rates=charging_rates,
    #                                                           algorithm_name=scheduling_alg.algorithm_name)
    #     self.assertTrue(all_energy_demands_met)
    #
    # def test_all3(self):
    #     start = datetime(2018, 4, 26, 0, 0, 0)
    #     end = datetime(2018, 4, 29, 23, 59, 59)
    #     period = 5
    #     # num_of_days = (end - start).days + 1
    #     filename = 'acndata_sessions_acn.json'
    #     evs, evs_time_not_normalised = get_evs_data_from_document(document=filename,
    #                                                               start=start,
    #                                                               end=end,
    #                                                               period=period,
    #                                                               amount_of_evs_interval=None)
    #
    #     available_energy_for_each_timestep = (
    #         bisection_search_Pt(EVs=evs,
    #                             start=start,
    #                             end=end,
    #                             period=period,
    #                             algorithm=SmoothedLeastLaxityAlg))
    #     scheduling_alg = SmoothedLeastLaxityAlg(EVs=evs,
    #                                             start=start,
    #                                             end=end,
    #                                             available_energy_for_each_timestep=available_energy_for_each_timestep,
    #                                             time_between_timesteps=period,
    #                                             cost_function=None)
    #     time_horizon = scheduling_alg.time_horizon
    #
    #     evs_remaining_energy_to_be_charged = []
    #     for ev in scheduling_alg.EVs:
    #         ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = ev
    #         evs_remaining_energy_to_be_charged.append(ev_requested_energy)
    #
    #
    #     charging_rates = scheduling_alg.charging_plan_for_all_ev
    #     for current_timestep in time_horizon:
    #         charging_rates, evs_remaining_energy_to_be_charged = (
    #             scheduling_alg.solve_for_current_timestep(current_timestep=current_timestep,
    #                                                       evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged))
    #
    #     charging_rates_within_bounds = check_charging_rates_within_bounds(evs=evs,
    #                                                                       charging_rates=charging_rates,
    #                                                                       )
    #     self.assertTrue(charging_rates_within_bounds)
    #
    #     infrastructure_not_violated = check_infrastructure_not_violated(charging_rates=charging_rates,
    #                                                                     available_energy_for_each_timestep=available_energy_for_each_timestep)
    #
    #     self.assertTrue(infrastructure_not_violated)
    #
    #     all_energy_demands_met = check_all_energy_demands_met(evs=evs,
    #                                                           charging_rates=charging_rates,
    #                                                           algorithm_name=scheduling_alg.algorithm_name)
    #     self.assertTrue(all_energy_demands_met)
