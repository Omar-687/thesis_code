
from datetime import datetime, timedelta

from opt_algorithm import OPT
from utils import *

import unittest
from os.path import exists





class testOPT(unittest.TestCase):




    # (24*12)/5
    def testEvsGeneration(self):
        filename = 'acndata_sessions_acn.json'
        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 26, 23, 59, 59)
        period = 5
        evs, evs_time_not_normalised = convert_to_evs(document=filename,
                                                      start=start,
                                                      end=end,
                                                      period=period,
                                                      amount_of_evs_interval=[5, 10])
        self.assertEqual(len(evs), len(evs_time_not_normalised), 'Length of evs list with normalised time should be same as length of evs list with not normalised time')
        self.assertTrue(5 <= len(evs) <= 10)
        arrival_time_index = 0
        departure_time_index = 1
        maximum_charging_rate_index = 2
        requested_energy_index = 3
        for i in range(len(evs)):
            self.assertEqual(len(evs[i]),4)
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
        evs, evs_time_not_normalised = convert_to_evs(document=filename,
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
        evs, evs_time_not_normalised = convert_to_evs(document=filename,
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





    def testinfeasiblesolution(self):
        ...
        # filename = 'acndata_sessions_acn.json'
        # start = datetime(2018, 4, 26, 0, 0, 0)
        # end = datetime(2018, 4, 26, 23, 59, 59)
        #
        # period = 5
        # evs, evs_time_not_normalised = convert_to_evs(document=filename,
        #                                               start=start,
        #                                               end=end,
        #                                               period=period,
        #                                               amount_of_evs_interval=[5, 10])
        #
        # save_evs_to_file(filename=txt_ev_info_filename,
        #                  evs=evs,
        #                  evs_with_time_not_normalised=evs_time_not_normalised)


    def test_basic_one_day_charging(self):

            # things to test
            # input to the program - evs data
            # output of the program - charging data
            
            filename = 'acndata_sessions_acn.json'
            txt_ev_info_filename = 'evs_data.txt'
            settings_filename = 'settings.txt'
            # f = open(filename)
            # data = json.load(f)
            start = datetime(2018, 4, 26, 0, 0, 0)
            end = datetime(2018, 4, 26, 23, 59,59)

            period = 5
            evs, evs_time_not_normalised = convert_to_evs(document=filename,
                           start=start,
                           end=end,
                           period=period,
                           amount_of_evs_interval=[5, 10])

            save_evs_to_file(filename=txt_ev_info_filename,
                             evs=evs,
                             evs_with_time_not_normalised=evs_time_not_normalised)

            available_energy_for_each_timestep = (
                bisection_search_Pt(EVs=evs,
                                    start=start,
                                    end=end,
                                    period=period,
                                    algorithm=OPT))

            # overenie ze sme dodali pozadovane mnozstvo energie vozidlam
            scheduling_alg = OPT(EVs=evs,
                                 start=start,
                                 end=end,
                                 available_energy_for_each_timestep=available_energy_for_each_timestep,
                                 time_between_timesteps=period,
                                 cost_function=None)
            feasibility, charging_rates = scheduling_alg.solve()
            cost_vector = scheduling_alg.get_cost_vector()
            maximum_charging_rate_index = 2
            requested_energy_index = 3
            # solver_tolerance_to_error = 1e-8
            solver_tolerance_to_error = 1e-5
            # solver_tolerance_to_error = 10e-8
            energy_charged = []
            # might have bigger error due to addition, more values etc therefore we use round for testing
            for i in range(len(charging_rates)):
                ev_requested_energy = evs[i][requested_energy_index]
                energy_charged.append(math.fsum(charging_rates[i]))
                accuracy = find_number_of_decimal_places(ev_requested_energy)
                self.assertTrue(round(energy_charged[i], accuracy) == ev_requested_energy)

            Pts = []
            for i in range(charging_rates.shape[1]):
                Pts.append(0)
                Pts[i] = math.fsum([available_energy_for_each_timestep, solver_tolerance_to_error])
                if math.fsum(charging_rates[:, i]) > Pts[i]:
                    a = 0
                    b = 0

                self.assertTrue(math.fsum(charging_rates[:, i]) <= Pts[i])

            # Pts = np.zeros(shape=(charging_rates.shape[1],)) + P_t + solver_tolerance_to_error

            # self.assertTrue(np.all(np.around(np.sum(charging_rates, axis=0),decimals=error_tol) <= Pts) == True)
            # self.assertTrue(np.all(np.sum(charging_rates, axis=0) <= Pts) == True)
            for i in range(len(charging_rates)):
                for j in range(len(charging_rates[i])):
                    # diff = abs(charging_rates[i][j] - evs[i][maximum_charging_rate_index])
                    self.assertTrue(0 <= charging_rates[i][j] <= evs[i][maximum_charging_rate_index] + solver_tolerance_to_error)

            create_settings_file(evs_num=len(evs),
                                 filename=settings_filename,
                                 start=start,
                                 end=end,
                                 P_t=available_energy_for_each_timestep,
                                 T=scheduling_alg.time_horizon,
                                 period=period,
                                 alg_name='OPT')

            # create_table(charging_profiles_matrix=charging_rates,
            #              charging_cost_vector=cost_vector,
            #              period=period,
            #              # P_t=P_t,
            #              update_after_hours=1,
            #              show_charging_costs=True


                         # )


    #     ran out of iterations error
    # TODO: find out why in general solver fails also why due to this error
    # 
    def test_more_days_charging(self):
        start = datetime(2018, 4, 26, 0, 0, 0)
        end = datetime(2018, 4, 29, 23, 59, 59)
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
                                algorithm=OPT))
        scheduling_alg = OPT(EVs=evs,
                             start=start,
                             end=end,
                             available_energy_for_each_timestep=available_energy_for_each_timestep,
                             time_between_timesteps=period,
                             cost_function=None)
        feasibility, charging_rates = scheduling_alg.solve()

        energy_charged = []
        maximum_charging_rate_index = 2
        requested_energy_index = 3
        # solver_tolerance_to_error = 1e-8
        solver_tolerance_to_error = 1e-5
        # solver_tolerance_to_error = 1e-6
        for i in range(len(evs)):
            ev_requested_energy = evs[i][requested_energy_index]
            energy_charged.append(math.fsum(charging_rates[i]))
            accuracy = find_number_of_decimal_places(ev_requested_energy)
            self.assertTrue(round(energy_charged[i], accuracy) == ev_requested_energy)

        Pts = []
        for i in range(charging_rates.shape[1]):
            Pts.append(0)
            Pts[i] = math.fsum([available_energy_for_each_timestep, solver_tolerance_to_error])
            if math.fsum(charging_rates[:, i]) > Pts[i]:
                a = 0
                b = 0
            self.assertTrue(math.fsum(charging_rates[:, i]) <= Pts[i])

        for i in range(len(charging_rates)):
            for j in range(len(charging_rates[i])):
                # diff = abs(charging_rates[i][j] - evs[i][maximum_charging_rate_index])
                self.assertTrue(0 <= charging_rates[i][j] <= evs[i][maximum_charging_rate_index] + solver_tolerance_to_error)
