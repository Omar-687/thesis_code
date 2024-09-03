from setuptools.command.alias import alias

from acnportal import acnsim, algorithms
import pytz
from datetime import datetime, timedelta
import random
import pandas as pd
import json
import math
from index_based_algs import *
from utils import *

import unittest
from os.path import exists

class testLLF(unittest.TestCase):
    def test1(self):
        filename = 'acndata_sessions_acn.json'
        txt_ev_info_filename = 'evs_data_file.txt'
        settings_filename = 'settings.txt'
        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 26, 23, 59, 59)

        period = 5
        evs, evs_time_not_normalised = convert_to_evs(document=filename,
                                                      start=start,
                                                      end=end,
                                                      period=period,
                                                      amount_of_evs_interval=[5, 10])

        save_evs_to_file(filename=txt_ev_info_filename,
                         evs=evs,
                         evs_with_time_not_normalised=evs_time_not_normalised)

        available_energy_for_each_timestep = bisection_search_Pt(EVs=evs,
                                  start=start,
                                  end=end,
                                  algorithm=LeastLaxityFirstAlg,
                                  period=period,
                                  error_tol=10e-4)

        scheduling_alg = (
            LeastLaxityFirstAlg(EVs=evs,
                                start=start,
                                end=end,
                                available_energy_for_each_timestep=available_energy_for_each_timestep,
                                time_between_timesteps=period,
                                cost_function=None))
        maximum_charging_rate_index = 2
        feasibility, charging_rates = scheduling_alg.solve()
        for i in range(len(charging_rates)):
            for j in range(len(charging_rates[i])):
                maximum_charging_rate = evs[i][maximum_charging_rate_index]
                self.assertTrue(charging_rates[i,j] >= 0)#
                self.assertTrue(maximum_charging_rate >= charging_rates[i,j])

        requested_energy_index = 3

        for i, ev in enumerate(evs):
            ev_requested_energy = evs[i][requested_energy_index]
            energy_charged = math.fsum(charging_rates[i])
            accuracy = find_number_of_decimal_places(ev_requested_energy)
            # doesnt seem to work
            # self.assertTrue(round(energy_charged, accuracy) == ev_requested_energy)
        Pts = []
        for i in range(charging_rates.shape[1]):
            Pts.append(0)

            Pts[i] = math.fsum([available_energy_for_each_timestep])
            if math.fsum(charging_rates[:, i]) > Pts[i]:
                a = 0
                b = 0
            self.assertTrue(math.fsum(charging_rates[:, i]) <= Pts[i])
        # create_settings_file(evs_num=len(evs),
        #                      filename=settings_filename,
        #                      start=start,
        #                      end=end,
        #                      P_t=available_energy_for_each_timestep,
        #                      T=scheduling_alg.time_horizon,
        #                      period=period,
        #                      alg_name='LLF')
        #
        # create_table(charging_profiles_matrix=charging_rates,
        #              charging_cost_vector=cost_vector,
        #              period=period,
        #              # P_t=P_t,
        #              update_after_hours=1,
        #              show_charging_costs=True
        #              # start=start,
        #              # end=end,
        #              # alg_name='OPT'
        #
        #              )





    def test2(self):
        # nefunguje
        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 29, 23, 59, 59)

        # start = datetime(2018, 4, 26, 0, 0, 0)
        # end = datetime(2018, 4, 26, 23, 59, 59)
        #
        # start = datetime(2018, 4, 26, 0, 0, 0)
        # end = datetime(2018, 4, 26, 23, 59, 59)

        period = 5
        # num_of_days = (end - start).days + 1
        filename = 'acndata_sessions_acn.json'
        evs, evs_time_not_normalised = convert_to_evs(document=filename,
                                                      start=start,
                                                      end=end,
                                                      period=period,
                                                      amount_of_evs_interval=None)

        available_energy_for_each_timestep = (
            bisection_search_Pt(EVs=evs,
                                start=start,
                                end=end,
                                period=period,
                                algorithm=LeastLaxityFirstAlg))
        scheduling_alg = LeastLaxityFirstAlg(EVs=evs,
                                                start=start,
                                                end=end,
                                                available_energy_for_each_timestep=available_energy_for_each_timestep,
                                                time_between_timesteps=period,
                                                cost_function=None)
        feasibility, charging_rates = scheduling_alg.solve()


        maximum_charging_rate_index = 2
        requested_energy_index = 3
        for i in range(len(evs)):
            ev_requested_energy = evs[i][requested_energy_index]
            energy_charged = math.fsum(charging_rates[i])
            accuracy = find_number_of_decimal_places(ev_requested_energy)
            self.assertTrue(round(energy_charged, accuracy) == ev_requested_energy)

        Pts = []
        for i in range(charging_rates.shape[1]):
            Pts.append(0)
            Pts[i] = math.fsum([available_energy_for_each_timestep])
            self.assertTrue(math.fsum(charging_rates[:, i]) <= Pts[i])

        for i in range(len(charging_rates)):
            for j in range(len(charging_rates[i])):
                # diff = abs(charging_rates[i][j] - evs[i][maximum_charging_rate_index])
                self.assertTrue(0 <= charging_rates[i][j])
                self.assertTrue(charging_rates[i][j] <= evs[i][maximum_charging_rate_index])

    def test3(self):
        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 29, 23, 59, 59)

        # start = datetime(2018, 4, 26, 0, 0, 0)
        # end = datetime(2018, 4, 26, 23, 59, 59)
        #
        # start = datetime(2018, 4, 26, 0, 0, 0)
        # end = datetime(2018, 4, 26, 23, 59, 59)

        period = 5
        # num_of_days = (end - start).days + 1
        filename = 'acndata_sessions_acn.json'
        evs, evs_time_not_normalised = convert_to_evs(document=filename,
                                                      start=start,
                                                      end=end,
                                                      period=period,
                                                      amount_of_evs_interval=None)

        available_energy_for_each_timestep = (
            bisection_search_Pt(EVs=evs,
                                start=start,
                                end=end,
                                period=period,
                                algorithm=LeastLaxityFirstAlg))
        scheduling_alg = LeastLaxityFirstAlg(EVs=evs,
                                                start=start,
                                                end=end,
                                                available_energy_for_each_timestep=available_energy_for_each_timestep,
                                                time_between_timesteps=period,
                                                cost_function=None)
        time_horizon = scheduling_alg.time_horizon
        evs_remaining_energy_to_be_charged = []
        for ev in scheduling_alg.EVs:
            ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = ev
            evs_remaining_energy_to_be_charged.append(ev_requested_energy)

        charging_rates = scheduling_alg.charging_plan_for_all_ev
        for current_timestep in time_horizon:
            charging_rates, evs_remaining_energy_to_be_charged =  scheduling_alg.solve_for_one_timestep(current_timestep=current_timestep,
                                                  evs_remaining_energy_to_be_charged=evs_remaining_energy_to_be_charged)

        


        maximum_charging_rate_index = 2
        requested_energy_index = 3
        for i in range(len(evs)):
            ev_requested_energy = evs[i][requested_energy_index]
            energy_charged = math.fsum(charging_rates[i])
            accuracy = find_number_of_decimal_places(ev_requested_energy)
            self.assertTrue(round(energy_charged, accuracy) == ev_requested_energy)

        Pts = []
        for i in range(charging_rates.shape[1]):
            Pts.append(0)
            Pts[i] = math.fsum([available_energy_for_each_timestep])
            self.assertTrue(math.fsum(charging_rates[:, i]) <= Pts[i])

        for i in range(len(charging_rates)):
            for j in range(len(charging_rates[i])):
                # diff = abs(charging_rates[i][j] - evs[i][maximum_charging_rate_index])
                self.assertTrue(0 <= charging_rates[i][j])
                self.assertTrue(charging_rates[i][j] <= evs[i][maximum_charging_rate_index])
