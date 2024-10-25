from utils import *
from preprocessing import (are_input_data_valid,
                           is_ev_valid,
                           are_dict_input_data_valid)
import unittest
from os.path import exists
from scheduling_alg import SchedulingAlg
from opt_algorithm import OPT
class testOtherMethods(unittest.TestCase):
    def testEvsGeneration(self):
        filename = 'acndata_sessions_acn.json'
        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 26, 23, 59, 59)
        period = 5
        evs, evs_time_not_normalised = get_evs_data_from_document(document=filename,
                                                                  start=start,
                                                                  end=end,
                                                                  period=period,
                                                                  amount_of_evs_interval=[5, 10])
        self.assertEqual(len(evs), len(evs_time_not_normalised),
                         'Length of evs list with normalised time should be same as length of evs list with not normalised time')
        self.assertTrue(5 <= len(evs) <= 10)
        arrival_time_index = 0
        departure_time_index = 1
        maximum_charging_rate_index = 2
        requested_energy_index = 3
        for i in range(len(evs)):
            self.assertEqual(len(evs[i]), 4)
            self.assertEqual(len(evs_time_not_normalised[i]), 4)
            self.assertTrue(evs[i][arrival_time_index] <= evs[i][departure_time_index])
            self.assertTrue(evs[i][requested_energy_index] >= 0)
            self.assertTrue(evs[i][maximum_charging_rate_index] >= 0)
            self.assertTrue(evs[i][arrival_time_index] >= 0)
            self.assertTrue(evs[i][departure_time_index] >= 0)
            self.assertTrue(end >= evs_time_not_normalised[i][arrival_time_index] >= start)
            self.assertTrue(end >= evs_time_not_normalised[i][departure_time_index] >= start)

    def testEvFileGeneration(self):
        filename = 'acndata_sessions_acn.json'
        txt_ev_info_filename = 'evs_data_test_file.txt'
        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 26, 23, 59, 59)

        period = 5
        evs, evs_time_not_normalised = get_evs_data_from_document(document=filename,
                                                                  start=start,
                                                                  end=end,
                                                                  period=period,
                                                                  amount_of_evs_interval=[5, 10])

        save_evs_to_file(filename=txt_ev_info_filename,
                         evs=evs,
                         evs_with_time_not_normalised=evs_time_not_normalised)

        file_exists = exists(txt_ev_info_filename)
        self.assertTrue(file_exists)

    def testEvFileContent(self):
        filename = 'acndata_sessions_acn.json'
        txt_ev_info_filename = 'evs_data_test_file.txt'
        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 26, 23, 59, 59)

        period = 5
        evs, evs_time_not_normalised = get_evs_data_from_document(document=filename,
                                                                  start=start,
                                                                  end=end,
                                                                  period=period,
                                                                  amount_of_evs_interval=[5, 10])

        save_evs_to_file(filename=txt_ev_info_filename,
                         evs=evs,
                         evs_with_time_not_normalised=evs_time_not_normalised)

        file_exists = exists(txt_ev_info_filename)

        f = open(txt_ev_info_filename, 'r')
        for index, line in enumerate(f.readline(), start=0):
            line = line.replace('\n', '')
            if index == 0:
                first_line = 'format i.auto: a_i, d_i, r_i, e_i'
                self.assertEqual(line, first_line)
            else:
                ...

    def test_wrong_evs(self):
        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 25, 23, 59, 59)
        evs = [[], 0, -10, {}, 'e', (7,),()]
        period = 5
        cost_function = SchedulingAlg.default_cost_function
        available_energy_for_each_timestep = 10
        for i in range(len(evs)):
            self.assertRaises(
                ValueError,
                OPT,
                EVs=evs[i],
                start=start,
                end=end,
                available_energy_for_each_timestep=available_energy_for_each_timestep,
                time_between_timesteps=period,
                cost_function=cost_function)

    def test_wrong_input_dates(self):
        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 25, 23, 59, 59)
        evs = [[1,2,3,4]]
        period = 5
        cost_function = SchedulingAlg.default_cost_function
        available_energy_for_each_timestep = 10
        self.assertRaises(
            ValueError,
            OPT,
            EVs=evs,
            start=start,
            end=end,
            available_energy_for_each_timestep=available_energy_for_each_timestep,
            time_between_timesteps=period,
            cost_function=cost_function)

        start = ['a', 0, -1, {}, [], ()]
        end = ['b', 10, 0, {'3':9}, [8], (4,)]
        for i in range(len(start)):
            self.assertRaises(
                ValueError,
                OPT,
                EVs=evs,
                start=start[i],
                end=end[i],
                available_energy_for_each_timestep=available_energy_for_each_timestep,
                time_between_timesteps=period,
                cost_function=cost_function)

    def test_wrong_period(self):
        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 26, 0, 0, 0)
        period = [-3, 0, (1, 2), [3, 4], {'a': 4}, 'e']
        evs = [[1,2,3,4]]
        cost_function = SchedulingAlg.default_cost_function
        available_energy_for_each_timestep = 10
        for i in range(len(period)):
            self.assertRaises(
                ValueError,
                OPT,
                EVs=evs,
                start=start,
                end=end,
                available_energy_for_each_timestep=available_energy_for_each_timestep,
                time_between_timesteps=period[i],
                cost_function=cost_function)
    def test_wrong_cost_set(self):
        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 26, 0, 0, 0)
        period = 10
        evs = [[1,2,3,4]]
        available_energy_for_each_timestep = 10
        cost_function = None
        self.assertRaises(
            ValueError,
            OPT,
            EVs=evs,
            start=start,
            end=end,
            available_energy_for_each_timestep=available_energy_for_each_timestep,
            time_between_timesteps=period,
            cost_function=cost_function)

        cost_function = object()
        manually_set_costs = object()
    #     TODO: finish the test
    def test_wrong_capacity(self):
        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 26, 23, 59, 59)
        evs = [[1,2,3,4]]
        period = 5
        cost_function = SchedulingAlg.default_cost_function
        available_energy_for_each_timestep = [-10, 0, [3, 4], {}, (5,), 'b']
        for i in range(len(available_energy_for_each_timestep)):
            self.assertRaises(
                ValueError,
                OPT,
                EVs=evs,
                start=start,
                end=end,
                available_energy_for_each_timestep=available_energy_for_each_timestep[i],
                cost_function=cost_function)

    def test_wrong_evse_number(self):
        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 26, 23, 59, 59)
        evs = []
        period = 5
        cost_function = SchedulingAlg.default_cost_function
        available_energy_for_each_timestep = 10
        number_of_evse = [0, -1, 0.5, 2.6, 'a', [3,4], {}, (5,)]
        for i in range(len(number_of_evse)):
            self.assertRaises(
                ValueError,
                OPT,
                EVs=evs,
                start=start,
                end=end,
                available_energy_for_each_timestep=available_energy_for_each_timestep,
                time_between_timesteps=period,
                number_of_evse=number_of_evse[i],
                cost_function=cost_function)







    def test_load_time_series(self):
        charging_networks = ['caltech', 'jpl', 'office_01']
        caltech_garages = ['California_Garage_01',
                           'California_Garage_02',
                           'LIGO_01',
                           'N_Wilson_Garage_01',
                           'S_Wilson_Garage_01']
        jpl_garages = ['Arroyo_Garage_01']
        office_01_garages = ['Parking_Lot_01']

        all_garages = [caltech_garages, jpl_garages, office_01_garages]

        # specified dates and charging stations in Learning-based Predictive Control via Real-time Aggregate Flexibility
        start_training_date = datetime(2018, 11, 1, tzinfo=timezone.utc)
        end_training_date = datetime(2019, 12, 1, tzinfo=timezone.utc)
        charging_network = charging_networks[0]
        garage = caltech_garages[0]
        period = 5
        start_testing_date = datetime(2019, 12, 2, tzinfo=timezone.utc)
        end_testing_date = datetime(2020, 1, 1, tzinfo=timezone.utc)

        for network_id in range(len(charging_networks)):
            for garage_id in range(len(all_garages[network_id])):
                garage = all_garages[network_id][garage_id]
                evs_data, evs_data_not_normalised_time = load_time_series_ev_data(
                    charging_network=charging_networks[network_id],
                    garages=[garage],
                    start=start_training_date,
                    end=end_training_date,
                    period=period)

                self.assertTrue(are_dict_input_data_valid(evs=evs_data,
                                                          start=start_training_date,
                                                          end=end_training_date))

                self.assertTrue(are_dict_input_data_valid(evs=evs_data_not_normalised_time,
                                                          start=start_training_date,
                                                          end=end_training_date))
                evs_data, evs_data_not_normalised_time = load_time_series_ev_data(
                    charging_network=charging_networks[network_id],
                    garages=[garage],
                    start=start_testing_date,
                    end=end_testing_date,
                    period=period)
                self.assertTrue(are_dict_input_data_valid(evs=evs_data,
                                                          start=start_testing_date,
                                                          end=end_testing_date))

                self.assertTrue(are_dict_input_data_valid(evs=evs_data_not_normalised_time,
                                                          start=start_testing_date,
                                                          end=end_testing_date))
    def test_charging_data_correctness(self):
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
        valid_timestamp_data = are_input_data_valid(evs=evs)
        valid_not_normalised_data = are_input_data_valid(evs=evs_time_not_normalised)
        self.assertTrue(valid_timestamp_data)
        self.assertTrue(valid_not_normalised_data)