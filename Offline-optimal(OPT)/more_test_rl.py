import numpy as np
import pytz
import random
import pandas as pd
import json
from opt_algorithm import OPT
import math
from sLLF_alg import SmoothedLeastLaxityAlg
from index_based_algs import *
from utils import *
from datetime import datetime, timedelta
# import ACNDataStatic
import unittest
from environment_ev import EVenvironment
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

#
# class TestRL(unittest.TestCase):
#     def test_1(self):
#         # test to generate a graph given in learning based control article
#         charging_networks = ['caltech', 'jpl', 'office_01']
#         # find which exact garage we will use
#
#         # 3 days should have at least 60 charging sessions out of 14
#         # possibly we need to choose all garages
#         caltech_garages = ['California_Garage_01',
#                            'California_Garage_02',
#                            'LIGO_01',
#                            'N_Wilson_Garage_01',
#                            'S_Wilson_Garage_01']
#         jpl_garages = ['Arroyo_Garage_01']
#         office_01_garages = ['Parking_Lot_01']
#
#         # start_testing = datetime(2019, 12, 2, 0, 0, 0,tzinfo=timezone.utc)
#         # end_testing = datetime(2020, 1, 1, 23, 59, 59,  tzinfo=timezone.utc)
#
#         start_testing = datetime(2020, 12, 2, 0, 0, 0, tzinfo=timezone.utc)
#         end_testing = datetime(2020, 12, 2, 23, 59, 59, tzinfo=timezone.utc)
#
#         # end_testing = datetime(2019, 12, 10, 23, 59, 59, tzinfo=timezone.utc)
#         ut_interval = [0 , 150]
#         maximum_charging_rate = 6.6
#         number_of_evse = 54
#         period = 12
#         number_of_evs_interval = [30, np.inf]
#         evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
#             charging_network=charging_networks[0],
#             # garages=caltech_garages,
#             garages=caltech_garages[:1],
#             start=start_testing,
#             end=end_testing,
#             period=period,
#             max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
#             number_of_evs_interval=number_of_evs_interval,
#             include_weekends=False,
#         )
#
#
#
#         #
#         # evs_timestamp_not_reset_list = convert_evs_diction_to_array(evs_diction=evs_timestamp_not_reset)
#
#         # should return error because more EVS than chargers
#         evs_timestamp_not_reset_list = create_dataset(arrival_timestamps=0,
#                                                       departure_timestamps=119,
#                                                       maximum_charging_rates=6.6,
#                                                       requested_energies=30,
#                                                       num_evs=54)
#         # cost_function = SchedulingAlg.inverse_cost_function
#         # cost_function = SchedulingAlg.unfair_cost_function
#         # cost_function = SchedulingAlg.default_cost_function
#
#         cost_file = '20241025-20241025 CAISO Day-Ahead Price (1).csv'
#         # TODO: pozriet sa na data lebo sa zda ze cerpam zo zlych dat
#
#         costs_loaded_manually = load_locational_marginal_prices(filename=cost_file, organization='VEA',period=period)
#
#         plot_costs(costs=costs_loaded_manually, period=period)
#         available_energy_for_each_timestep = 1000
#         scheduling_alg = OPT(EVs=evs_timestamp_not_reset_list,
#                              start=start_testing,
#                              end=end_testing,
#                              available_energy_for_each_timestep=available_energy_for_each_timestep,
#                              ut_interval=ut_interval,
#                              time_between_timesteps=period,
#                              number_of_evse=number_of_evse,
#                              costs_loaded_manually=costs_loaded_manually
#                              # cost_function=cost_function
#
#                              )
#         feasibility, charging_rates = scheduling_alg.solve()
#         cost_vector = scheduling_alg.cost_vector
#
#         # cost_vector = scheduling_alg.get_cost_vector()
#
#         charging_rates_within_bounds = check_charging_rates_within_bounds(evs=evs_timestamp_not_reset_list,
#                                                                           charging_rates=charging_rates,
#                                                                           )
#         self.assertTrue(charging_rates_within_bounds)
#
#         infrastructure_not_violated = check_infrastructure_not_violated(charging_rates=charging_rates,
#                                                                         available_energy_for_each_timestep=scheduling_alg.available_energy_for_each_timestep)
#
#         self.assertTrue(infrastructure_not_violated)
#
#         all_energy_demands_met = check_all_energy_demands_met(evs=evs_timestamp_not_reset_list,
#                                                               charging_rates=charging_rates,
#                                                               algorithm_name=scheduling_alg.algorithm_name)
#         self.assertTrue(all_energy_demands_met)
#
#         draw_barchart_sessions(schedule=charging_rates, evs_dict_reseted=evs_timestamp_reset)
#         mpe_values = []
#         costs = []
#
#         given_day = start_testing
#         i = 0
#         while given_day <= end_testing:
#             timesteps_of_one_day = int((24*60)/period)
#             start_of_day_timesteps = i*timesteps_of_one_day
#             end_of_day_timesteps = (i + 1)*timesteps_of_one_day
#             schedule_for_given_day = charging_rates[:,start_of_day_timesteps:end_of_day_timesteps]
#             # evs_for_given_day = evs_timestamp_reset[given_day]
#             evs_for_given_day = evs_timestamp_not_reset_list
#
#             costs_for_given_day = calculate_cumulative_costs(schedule=schedule_for_given_day,
#                                                              cost_vector=scheduling_alg.cost_vector)
#
#             charging_in_time_graph(ut_signals_offline=get_ut_signals_from_schedule(schedule=schedule_for_given_day),period=period)
#             mpe_values.append(mpe_error_fun(schedule=schedule_for_given_day, evs=evs_for_given_day))
#             costs.append(costs_for_given_day)
#             given_day += timedelta(days=1)
#             i += 1
#             # mpe = mpe_error_fun()
#             # mpe_values.append()
#         mpe_per_day_graph(mpe_values=mpe_values)
#         costs_per_day_graph(costs=costs)
#         mpe_cost_graph(mpe_values=mpe_values, cost_values=mpe_values)

    # # seems matching with ACN Data book stats
    # def test1_for_test_days(self):
    #     charging_networks = ['caltech', 'jpl', 'office_01']
    #     # find which exact garage we will use
    #
    #     # 3 days should have at least 60 charging sessions out of 14
    #     # possibly we need to choose all garages
    #     caltech_garages = ['California_Garage_01',
    #                        'California_Garage_02',
    #                        'LIGO_01',
    #                        'N_Wilson_Garage_01',
    #                        'S_Wilson_Garage_01']
    #     jpl_garages = ['Arroyo_Garage_01']
    #     office_01_garages = ['Parking_Lot_01']
    #
    #     # start_testing = datetime(2019, 12, 2, 0, 0, 0,tzinfo=timezone.utc)
    #     # end_testing = datetime(2020, 1, 1, 23, 59, 59,  tzinfo=timezone.utc)
    #
    #     start_testing = datetime(2018, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
    #     end_testing = datetime(2018, 11, 1, 23, 59, 59, tzinfo=timezone.utc)
    #
    #     # end_testing = datetime(2019, 12, 10, 23, 59, 59, tzinfo=timezone.utc)
    #     ut_interval = [0, 150]
    #     maximum_charging_rate = 6.6
    #     number_of_evse = 54
    #     period = 12
    #     number_of_evs_interval = [0, np.inf]
    #     evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
    #         charging_network=charging_networks[0],
    #         # garages=caltech_garages,
    #         garages=caltech_garages[:1],
    #         start=start_testing,
    #         end=end_testing,
    #         period=period,
    #         max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
    #         number_of_evs_interval=number_of_evs_interval,
    #         include_weekends=True,
    #
    #     )
    #
    #     a = 0
    #
    # # seems ok as well matching with acndata article
    # def test1_for_test_days2(self):
    #     charging_networks = ['caltech', 'jpl', 'office_01']
    #     # find which exact garage we will use
    #
    #     # 3 days should have at least 60 charging sessions out of 14
    #     # possibly we need to choose all garages
    #     caltech_garages = ['California_Garage_01',
    #                        'California_Garage_02',
    #                        'LIGO_01',
    #                        'N_Wilson_Garage_01',
    #                        'S_Wilson_Garage_01']
    #     jpl_garages = ['Arroyo_Garage_01']
    #     office_01_garages = ['Parking_Lot_01']
    #
    #     # start_testing = datetime(2019, 12, 2, 0, 0, 0,tzinfo=timezone.utc)
    #     # end_testing = datetime(2020, 1, 1, 23, 59, 59,  tzinfo=timezone.utc)
    #
    #     start_testing = datetime(2019, 11, 1, 0, 0, 0, tzinfo=timezone.utc)
    #     end_testing = datetime(2019, 12, 1, 23, 59, 59, tzinfo=timezone.utc)
    #
    #     # end_testing = datetime(2019, 12, 10, 23, 59, 59, tzinfo=timezone.utc)
    #     ut_interval = [0, 150]
    #     maximum_charging_rate = 6.6
    #     number_of_evse = 54
    #     period = 12
    #     number_of_evs_interval = [0, np.inf]
    #     evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
    #         charging_network=charging_networks[0],
    #         # garages=caltech_garages,
    #         garages=caltech_garages[:1],
    #         start=start_testing,
    #         end=end_testing,
    #         period=period,
    #         max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
    #         number_of_evs_interval=number_of_evs_interval,
    #         include_weekends=True,
    #
    #     )
    #     a = 0
    #
    # def test1_for_test_days3(self):
    #     charging_networks = ['caltech', 'jpl', 'office_01']
    #     # find which exact garage we will use
    #
    #     # 3 days should have at least 60 charging sessions out of 14
    #     # possibly we need to choose all garages
    #     caltech_garages = ['California_Garage_01',
    #                        'California_Garage_02',
    #                        'LIGO_01',
    #                        'N_Wilson_Garage_01',
    #                        'S_Wilson_Garage_01']
    #     jpl_garages = ['Arroyo_Garage_01']
    #     office_01_garages = ['Parking_Lot_01']
    #
    #     # start_testing = datetime(2019, 12, 2, 0, 0, 0,tzinfo=timezone.utc)
    #     # end_testing = datetime(2020, 1, 1, 23, 59, 59,  tzinfo=timezone.utc)
    #
    #     # Define the Los Angeles timezone
    #     la_tz = pytz.timezone('America/Los_Angeles')
    #
    #     # start_testing = datetime(2019, 11, 2, 0, 0, 0, tzinfo=timezone.utc)
    #     # end_testing = datetime(2020, 1, 1, 23, 59, 59, tzinfo=timezone.utc)
    #
    #     start_testing = la_tz.localize(datetime(2019, 12, 2, 0, 0, 0))
    #     end_testing = la_tz.localize(datetime(2020, 1, 1, 23, 59, 59))
    #
    #     # end_testing = datetime(2019, 12, 10, 23, 59, 59, tzinfo=timezone.utc)
    #     ut_interval = [0, 150]
    #     maximum_charging_rate = 6.6
    #     number_of_evse = 54
    #     period = 12
    #     number_of_evs_interval = [0, np.inf]
    #     evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
    #         charging_network=charging_networks[0],
    #         # garages=caltech_garages,
    #         garages=caltech_garages,
    #         start=start_testing,
    #         end=end_testing,
    #         period=period,
    #         max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
    #         number_of_evs_interval=number_of_evs_interval,
    #         include_weekends=False,
    #
    #     )
    #     a = 0
    #     given_day = start_testing
    #     while given_day <= end_testing:
    #         if evs_timestamp_reset.get(given_day, None) is None:
    #             given_day += timedelta(days=1)
    #             continue
    #         evs_for_given_day = evs_timestamp_reset[given_day]
    #         plot_arrivals_for_given_day(evs=evs_for_given_day,day=given_day,period=period)
    #         plot_departures_for_given_day(evs=evs_for_given_day,day=given_day,period=period)
    #         given_day += timedelta(days=1)
    #
    #
    # def test_2(self):
    #     mse_values = []
    #     for i in range(10):
    #         mse_values.append(random.uniform(0, 1))
    #     mse_per_day_graph(mse_values=mse_values)
    #
    # def test_environment_bad(self):
    #     scheduling_algorithm = LeastLaxityFirstAlg
    #     cost_list = None
    #     charging_days_list = None
    #     environ = EVenvironment(scheduling_algorithm=scheduling_algorithm,
    #                             cost_list=cost_list,
    #                             charging_days_list=charging_days_list)
    #     action = [np.random.uniform(0,1) for i in range(10)]
    #
    #     evs_timestamp_not_reset_list = create_dataset(arrival_timestamps=1,
    #                                                   departure_timestamps=119,
    #                                                   maximum_charging_rates=6.6,
    #                                                   requested_energies=30,
    #                                                   num_evs=60)
    #     environ.charging_data = evs_timestamp_not_reset_list
    #
    #     environ.step(action=action)
    #
    #
    # def test_environment_good(self):
    #     scheduling_algorithm = LeastLaxityFirstAlg
    #     cost_list = None
    #     charging_days_list = None
    #     environ = EVenvironment(scheduling_algorithm=scheduling_algorithm,
    #                             cost_list=cost_list,
    #                             charging_days_list=charging_days_list)
    #     action = [np.random.uniform(0,1) for i in range(10)]
    #
    #     evs_timestamp_not_reset_list = create_dataset(arrival_timestamps=1,
    #                                                   departure_timestamps=119,
    #                                                   maximum_charging_rates=6.6,
    #                                                   requested_energies=30,
    #                                                   num_evs=54)
    #     environ.charging_data = evs_timestamp_not_reset_list
    #
    #     environ.step(action=action)
    #
    #
    # def test_prices_good(self):
    #     a = 0
    #     cost_file = '20241025-20241025 CAISO Day-Ahead Price (1).csv'
    #     cost_vector = load_locational_marginal_prices(filename=cost_file, organization=None, period=12)
    # # OK
    # def test_3(self):
    #     filename = 'rl-hyperparameters.json'
    #     input_dict = {}
    #     save_hyperparam_into_file(input_diction=input_dict,filename=filename)

        # cost_file = '20241025-20241025 CAISO Day-Ahead Price (1).csv'
        # cost_vector = load_locational_marginal_prices(filename=cost_file, organization=None, period=12)