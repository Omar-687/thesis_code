
from datetime import datetime, timedelta, timezone

from opt_algorithm import OPT
from utils import *

import unittest
from os.path import exists
from testing_functions import (check_all_energy_demands_met,
                               check_infrastructure_not_violated,
                               check_charging_rates_within_bounds)

from preprocessing import (are_input_data_valid,
                           is_ev_valid,
                           are_dict_input_data_valid)

from scheduling_alg import SchedulingAlg


# TODO: test that algorithms work for different cost functions
# TODO: check how  tongxin li implemented MPC with chargers (how did he manage them) and add specified number of EVSE


class testOPT(unittest.TestCase):




    # (24*12)/5





    def test_basic_one_day_charging(self):
        filename = 'acndata_sessions_acn.json'
        txt_ev_info_filename = 'evs_data.txt'
        settings_filename = 'settings.txt'
        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 26, 23, 59,59)
        period = 5
        evs, evs_time_not_normalised = get_evs_data_from_document(document=filename,
                                                                  start=start,
                                                                  end=end,
                                                                  period=period,
                                                                  amount_of_evs_interval=[5, 10])
        cost_function = SchedulingAlg.default_cost_function
        number_of_evse = 54
        save_evs_to_file(filename=txt_ev_info_filename,
                         evs=evs,
                         evs_with_time_not_normalised=evs_time_not_normalised)

        available_energy_for_each_timestep = (
            bisection_search_Pt(EVs=evs,
                                start=start,
                                end=end,
                                period=period,
                                algorithm=OPT,
                                cost_function=cost_function))

        scheduling_alg = OPT(EVs=evs,
                             start=start,
                             end=end,
                             available_energy_for_each_timestep=available_energy_for_each_timestep,
                             time_between_timesteps=period,
                             cost_function=cost_function)
        feasibility, charging_rates = scheduling_alg.solve()
        cost_vector = scheduling_alg.cost_vector

        # cost_vector = scheduling_alg.get_cost_vector()

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

        create_settings_file(evs_num=len(evs),
                             filename=settings_filename,
                             start=start,
                             end=end,
                             available_energy_for_each_timestep=available_energy_for_each_timestep,
                             time_horizon=scheduling_alg.time_horizon,
                             period=period,
                             number_of_evse=number_of_evse,
                             charging_network=[],
                             garages=[],
                             algorithm_name='OPT')

        create_table(charging_profiles_matrix=charging_rates,
                     charging_cost_vector=cost_vector,
                     period=period,
                     show_charging_costs=True)

    def test_basic_one_day_charging_with_time_series(self):
        charging_networks = ['caltech', 'jpl', 'office_01']
        caltech_garages = ['California_Garage_01',
                           'California_Garage_02',
                           'LIGO_01',
                           'N_Wilson_Garage_01',
                           'S_Wilson_Garage_01']
        jpl_garages = ['Arroyo_Garage_01']
        office_01_garages = ['Parking_Lot_01']
        start = datetime(2019, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2019, 1, 3, 0, 0, 0,tzinfo=timezone.utc)
        period = 5

        cost_function = SchedulingAlg.default_cost_function

        evs, evs_time_not_normalised = load_time_series_ev_data(
            charging_network=charging_networks[0],
            garages=[caltech_garages[0]],
            start=start,
            end=end,
            period=period,
            reset_timestamp_after_each_day=False,
            include_weekends=True,
            include_days_with_less_than_30_charging_sessions=True
        )


        save_evs_to_file(filename=txt_ev_info_filename,
                         evs=evs,
                         evs_with_time_not_normalised=evs_time_not_normalised)


        evs = convert_dict_evs_to_list_evs(evs=evs)
        available_energy_for_each_timestep = (
            bisection_search_Pt(EVs=evs,
                                start=start,
                                end=end,
                                period=period,
                                algorithm=OPT,
                                cost_function=cost_function))

        scheduling_alg = OPT(EVs=evs,
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





    def test_more_days_charging(self):
        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 29, 23, 59, 59)
        period = 5
        # num_of_days = (end - start).days + 1
        filename = 'acndata_sessions_acn.json'
        evs, evs_time_not_normalised = get_evs_data_from_document(document=filename,
                                                                  start=start,
                                                                  end=end,
                                                                  period=period,
                                                                  amount_of_evs_interval=None)

        cost_function = SchedulingAlg.default_cost_function
        number_of_evse = 54
        available_energy_for_each_timestep = (
            bisection_search_Pt(EVs=evs,
                                start=start,
                                end=end,
                                period=period,
                                number_of_evse=number_of_evse,
                                algorithm=OPT,
                                cost_function=cost_function
                                ))
        scheduling_alg = OPT(EVs=evs,
                             start=start,
                             end=end,
                             available_energy_for_each_timestep=available_energy_for_each_timestep,
                             time_between_timesteps=period,
                             number_of_evse=number_of_evse,
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


