from sLLF_alg import SmoothedLeastLaxityAlg
# import ACNDataStatic
import unittest

from index_based_algs import *
from sLLF_alg import SmoothedLeastLaxityAlg
from testing_functions import (check_all_energy_demands_met,
                               check_charging_rates_within_bounds,
                               check_infrastructure_not_violated)
from utils import *


class testsLLF(unittest.TestCase):
    def test1(self):
        filename = 'caiso_2016/acndata_sessions_acn.json'
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

        # available_energy_for_each_timestep = 1000

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
                             charging_networks_chosen=['caltech'],
                             garages_chosen=caltech_garages,
                             number_of_evse=number_of_evse)

        create_table(charging_profiles_matrix=charging_rates,
                     charging_cost_vector=cost_vector,
                     period=period,
                     show_charging_costs=True)
    def test_with_timeseries_data(self):
        charging_networks = ['caltech', 'jpl', 'office_01']
        caltech_garages = ['California_Garage_01',
                           'California_Garage_02',
                           'LIGO_01',
                           'N_Wilson_Garage_01',
                           'S_Wilson_Garage_01']
        jpl_garages = ['Arroyo_Garage_01']
        office_01_garages = ['Parking_Lot_01']
        start = datetime(2019, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2019, 1, 20, 0, 0, 0, tzinfo=timezone.utc)
        period = 5
        number_of_evse = 54
        charging_network = [charging_networks[0]]
        garages = [caltech_garages[0], caltech_garages[1]]
        cost_function = SchedulingAlg.default_cost_function

        evs, evs_time_not_normalised = load_time_series_ev_data(
            charging_network=charging_network[0],
            garages=garages,
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
        # available_energy_for_each_timestep = (
        #     bisection_search_Pt(EVs=evs,
        #                         start=start,
        #                         end=end,
        #                         period=period,
        #                         number_of_evse=number_of_evse,
        #                         algorithm=SmoothedLeastLaxityAlg,
        #                         cost_function=cost_function))
        available_energy_for_each_timestep = 1000
        scheduling_alg = SmoothedLeastLaxityAlg(EVs=evs,
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

        create_settings_file(evs_num=len(evs),
                             filename=settings_filename,
                             start=start,
                             end=end,
                             available_energy_for_each_timestep=available_energy_for_each_timestep,
                             time_horizon=scheduling_alg.time_horizon,
                             period=period,
                             algorithm_name=scheduling_alg.algorithm_name,
                             charging_networks_chosen=charging_network,
                             garages_chosen=garages,
                             number_of_evse=number_of_evse)

        #
    def test2(self):
        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 29, 23, 59, 59)
        period = 5
        filename = 'caiso_2016/acndata_sessions_acn.json'
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
        filename = 'caiso_2016/acndata_sessions_acn.json'
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
