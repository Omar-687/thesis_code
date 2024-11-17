import numpy as np
import pytz
from datetime import datetime, timedelta
import random
import pandas as pd
import json
import math

from torch.profiler import schedule

from sLLF_alg import SmoothedLeastLaxityAlg
from index_based_algs import *
from utils import *
from opt_algorithm import OPT
from stable_baselines3.common.evaluation import evaluate_policy
# import ACNDataStatic
# from stable_baselines3.gail import ExpertDataset
from environment_ev import EVenvironment
from testing_functions import (check_all_energy_demands_met,
                               check_charging_rates_within_bounds,
                               check_infrastructure_not_violated)
import unittest
from os.path import exists
from stable_baselines3 import A2C, SAC, PPO
import gymnasium
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure


# TODO: try just training and testing if it will work on that environemtn 
# look up the documentation on sb3
# env = EVenvironment()
# # Here the expert is a random agent
# # but it can be any python function, e.g. a PID controller
# def dummy_expert(_obs):
#     """
#     Random agent. It samples actions randomly
#     from the action space of the environment.
#
#     :param _obs: (np.ndarray) Current observation
#     :return: (np.ndarray) action taken by the expert
#     """
#     return env.action_space.sample()
# generate_expert_traj(dummy_expert, 'dummy_expert_cartpole', env, n_episodes=10)


class FigureRecorderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        # Plot values (here a random variable)
        figure = plt.figure()
        figure.add_subplot().plot(np.random.random(3))
        # Close the figure after logging it
        self.logger.record("trajectory/figure", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
        plt.close()
        return True



class moretestsRL(unittest.TestCase):
    charging_networks = ['caltech', 'jpl', 'office_01']
    # find which exact garage we will use

    # 3 days should have at least 60 charging sessions out of 14
    # possibly we need to choose all garages
    caltech_garages = ['California_Garage_01',
                       'California_Garage_02',
                       'LIGO_01',
                       'N_Wilson_Garage_01',
                       'S_Wilson_Garage_01']
    jpl_garages = ['Arroyo_Garage_01']
    office_01_garages = ['Parking_Lot_01']
    def test_random_agent_test(self):
        scheduling_algorithm = SmoothedLeastLaxityAlg
        charging_days_list = None
        period = 12

        cost_file = '20241025-20241025 CAISO Day-Ahead Price (1).csv'
        cost_list = load_locational_marginal_prices(filename=cost_file, organization=None, period=period)
        env = EVenvironment(scheduling_algorithm=scheduling_algorithm,
                            cost_list=cost_list,
                            charging_days_list=charging_days_list)
        evs_timestamp_not_reset_list = create_dataset(arrival_timestamps=1,
                                                      departure_timestamps=119,
                                                      maximum_charging_rates=6.6,
                                                      requested_energies=30,
                                                      num_evs=54)
        options = {'charging_data': evs_timestamp_not_reset_list}
        env.charging_data = evs_timestamp_not_reset_list
        obs, info = env.reset(options=options)
        n_steps = 10
        for _ in range(n_steps):
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                obs, info = env.reset(options=options)
    def test_saving_model(self):
        dir_where_saved_models_are = 'SavedModels/'
        env_id = "CartPole-v1"
        ppo_expert = PPO("MlpPolicy", env_id, verbose=1)
        ppo_expert.learn(total_timesteps=int(3e4))
        filename = dir_where_saved_models_are + 'ppo_info.json'
        input_diction = {}
        save_hyperparam_into_file(filename, input_diction)

        ppo_expert.save(dir_where_saved_models_are+"ppo_expert")

    def test_check_ev_environment_validity(self):
        la_tz = pytz.timezone('America/Los_Angeles')
        start_testing = la_tz.localize(datetime(2019, 12, 16, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2019, 12, 16, 23, 59, 59))

        maximum_charging_rate = 6.6
        period = 12
        number_of_evs_interval = [30, np.inf]
        cost_file = '20241025-20241025 CAISO Day-Ahead Price (1).csv'
        evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
            charging_network=charging_networks[0],
            # garages=caltech_garages,
            garages=caltech_garages[:1],
            start=start_testing,
            end=end_testing,
            period=period,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
        )


        cost_list =  load_locational_marginal_prices(filename=cost_file, organization=None, period=period)
        charging_days_list = list(evs_timestamp_reset.keys())
        scheduling_algorithm = LeastLaxityFirstAlg
        # scheduling_algorithm = SmoothedLeastLaxityAlg
        env = EVenvironment(scheduling_algorithm, charging_days_list, cost_list)
        # env.reset(seed=0)
        check_env(env, warn=True)


    def test_environment_bad(self):

        scheduling_algorithm = LeastLaxityFirstAlg
        cost_list = None
        charging_days_list = None
        environ = EVenvironment(scheduling_algorithm=scheduling_algorithm,
                                cost_list=cost_list,
                                charging_days_list=charging_days_list)
        action = [np.random.uniform(0,1) for i in range(10)]

        evs_timestamp_not_reset_list = create_dataset(arrival_timestamps=1,
                                                      departure_timestamps=119,
                                                      maximum_charging_rates=6.6,
                                                      requested_energies=30,
                                                      num_evs=60)
        environ.charging_data = evs_timestamp_not_reset_list

        environ.step(action=action)

    def test_pretrain_agent(self):
        ...


    def test_env_step(self):
        # scheduling_algorithm = LeastLaxityFirstAlg

        scheduling_algorithm = SmoothedLeastLaxityAlg

        cost_list = [0, 0.1]
        charging_days_list = None
        environ = EVenvironment(scheduling_algorithm=scheduling_algorithm,
                                cost_list=cost_list,
                                charging_days_list=charging_days_list)
        action = [np.random.uniform(0,1) for i in range(10)]

        evs_timestamp_not_reset_list = create_dataset(arrival_timestamps=1,
                                                      departure_timestamps=119,
                                                      maximum_charging_rates=6.6,
                                                      requested_energies=30,
                                                      num_evs=54)
        options = {'charging_data': evs_timestamp_not_reset_list}
        environ.charging_data = evs_timestamp_not_reset_list
        # reset is first called
        environ.reset(options=options)
        environ.signal_ut = 30
        environ.step(action=action)


    def test_env_reset(self):
        la_tz = pytz.timezone('America/Los_Angeles')
        start_testing = la_tz.localize(datetime(2019, 12, 2, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2020, 1, 1, 23, 59, 59))

        ut_interval = [0, 150]
        maximum_charging_rate = 6.6
        number_of_evse = 54
        period = 12
        number_of_evs_interval = [30, np.inf]
        evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
            charging_network=charging_networks[0],
            # garages=caltech_garages,
            garages=caltech_garages[:1],
            start=start_testing,
            end=end_testing,
            period=period,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
        )
        charging_days_list = list(evs_timestamp_reset.keys())
        scheduling_algorithm = LeastLaxityFirstAlg
        cost_list = [0, 0.1]
        environ = EVenvironment(scheduling_algorithm=scheduling_algorithm,
                                cost_list=cost_list,
                                charging_days_list=charging_days_list)
        environ.reset()

    def testRLsaclibrary(self):
        env = gymnasium.make("Pendulum-v1", render_mode="human")

        model = SAC("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000, log_interval=4)
        model.save("sac_pendulum")

        del model  # remove to demonstrate saving and loading

        model = SAC.load("sac_pendulum")

        obs, info = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
    def testRLlibrary(self):
        model = SAC("MlpPolicy", "Pendulum-v1", tensorboard_log="./sac/", verbose=1,
                    gamma=0.5)
        reward_function_form = ''
        model.learn(50000, callback=TensorboardCallback())
    # more less the same, only difference with 1-2 evs a day and requested energies are similar as well there is not big difference
    def test_load_json_file_advanced(self):
        document = 'acndata_sessions_testing.json'
        la_tz = pytz.timezone('America/Los_Angeles')
        # testing phase
        # data should be ok, offseted time is prior to oct 2019
        start_testing = la_tz.localize(datetime(2019, 12, 2, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2020, 1, 1, 23, 59, 59))
        number_of_evs_interval = [30, np.inf]
        period = 12
        maximum_charging_rate = 6.6
        evs_timestamp_reset, evs_timestamp_reset, evs_time_not_normalised = get_evs_data_from_document_advanced_settings(
            document=document,
            start=start_testing,
            end=end_testing,
            period=period,
            allow_overday_charging=False,
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False)
        a = 0

        evs_timestamp_reset1, evs_timestamp_not_reset1, evs_time_not_normalised_time_reset1 = load_time_series_ev_data(
            charging_network=charging_networks[0],
            # garages=caltech_garages,
            garages=caltech_garages,
            start=start_testing,
            end=end_testing,
            period=period,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
            include_overday_charging=False
        )
        sum_of_charging = {}
        sum_of_charging_time_series = {}
        for key,value in evs_timestamp_reset.items():
            sum_of_charging[key] = 0

            for v in value:
                sum_of_charging[key] += v[-1]

        for key, value in evs_timestamp_reset1.items():
            sum_of_charging_time_series[key] = 0

            for v in value:
                sum_of_charging_time_series[key] += v[-1]
        b = 0

    def test_load_data_without_overday_charging(self):
        la_tz = pytz.timezone('America/Los_Angeles')
        # testing phase
        start_testing = la_tz.localize(datetime(2019, 12, 2, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2020, 1, 1, 23, 59, 59))
        maximum_charging_rate = 6.6
        period = 12
        number_of_evse = 54
        number_of_evs_interval = [0, np.inf]
        # this data must be loaded even if environment loads data separately, to filter charging days
        evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
            charging_network=charging_networks[0],
            # garages=caltech_garages,
            garages=caltech_garages,
            start=start_testing,
            end=end_testing,
            period=period,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
            include_overday_charging=True
        )


        evs_timestamp_reset1, evs_timestamp_not_reset1, evs_time_not_normalised_time_reset1 = load_time_series_ev_data(
            charging_network=charging_networks[1],
            # garages=caltech_garages,
            garages=jpl_garages,
            start=start_testing,
            end=end_testing,
            period=period,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=True,
            include_overday_charging=True

        )

        evs_timestamp_reset2, evs_timestamp_not_reset2, evs_time_not_normalised_time_reset2 = load_time_series_ev_data(
            charging_network=charging_networks[2],
            # garages=caltech_garages,
            garages=office_01_garages,
            start=start_testing,
            end=end_testing,
            period=period,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=True,
            include_overday_charging=True

        )
        a = 0
        # values = evs_timestamp_reset['2019-12-16 00:00:00-08:00']
        sum_of_charging = 0
        for v in evs_timestamp_reset[list(evs_timestamp_reset.keys())[11]]:
            sum_of_charging += v[-1]
    def test_jpl_error(self):
        la_tz = pytz.timezone('America/Los_Angeles')
        # testing phase
        start_testing = la_tz.localize(datetime(2019, 12, 23, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2019, 12, 23, 23, 59, 59))
        # possibly necessary to increase power limit and power levels due to intensity
        maximum_charging_rate = 6.8
        time_between_timesteps = 12
        number_of_evse = 54
        number_of_evs_interval = [30, np.inf]
        document = 'acndata_sessions_jpl.json'
        # this data must be loaded even if environment loads data separately, to filter charging days
        evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised = (
            get_evs_data_from_document_advanced_settings(document=document,
                                                         start=start_testing,
                                                         end=end_testing,
                                                         number_of_evs_interval=number_of_evs_interval,
                                                         include_weekends=False,
                                                         allow_overday_charging=False,
                                                         period=time_between_timesteps,
                                                         max_charging_rate_within_interval=[maximum_charging_rate,maximum_charging_rate]))

        cost_file = '20241025-20241025 CAISO Day-Ahead Price (1).csv'
        # costs_loaded_manually = load_locational_marginal_prices(filename=cost_file, organization='VEA',period=period)
        cost_per_hour_man, costs_loaded_manually = load_locational_marginal_prices(filename=cost_file,
                                                                                   organization=None,
                                                                                       period=time_between_timesteps)
        given_gamma = 1
        power_limit = 300
        ev = evs_timestamp_reset[start_testing][61]
        ev[0] = 0
        # 61 element causes problems
        scheduling_alg = OPT(EVs=[evs_timestamp_reset[start_testing][61]],
                             start=start_testing,
                             end=end_testing,
                             gamma=given_gamma,
                             power_limit=power_limit,
                             time_between_timesteps=time_between_timesteps,
                             number_of_evse=number_of_evse,
                             costs_loaded_manually=costs_loaded_manually)
        feasibility, solution = scheduling_alg.solve(verbose=True)
    # stop doing jpl further for now, there is no need to check signals because they might be bit more inaccurate
    def test_jpl_charging_days(self):
        la_tz = pytz.timezone('America/Los_Angeles')
        # testing phase
        start_testing = la_tz.localize(datetime(2019, 12, 2, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2020, 1, 1, 23, 59, 59))
        # possibly necessary to increase power limit and power levels due to intensity
        # depending on the data max charging rate will have to be increased
        maximum_charging_rate = 7
        time_between_timesteps = 12
        number_of_evse = 54
        number_of_evs_interval = [30, np.inf]
        document = 'acndata_sessions_jpl.json'
        # this data must be loaded even if environment loads data separately, to filter charging days
        evs_timestamp_reset,evs_timestamp_not_reset, evs_time_not_normalised = (
            get_evs_data_from_document_advanced_settings(document=document,
                                                         start=start_testing,
                                                         end=end_testing,
                                                         number_of_evs_interval=number_of_evs_interval,
                                                         include_weekends=False,
                                                         allow_overday_charging=False,
                                                         period=time_between_timesteps,
                                                         max_charging_rate_within_interval=[maximum_charging_rate,maximum_charging_rate]))

        cost_file = '20241025-20241025 CAISO Day-Ahead Price (1).csv'
        # costs_loaded_manually = load_locational_marginal_prices(filename=cost_file, organization='VEA',period=period)
        cost_per_hour_man, costs_loaded_manually = load_locational_marginal_prices(filename=cost_file,
                                                                                       organization=None, period=time_between_timesteps)

        path_to_outer_directory = fr'testing_days_jpl/'
        cost_plot_file = 'cost_function.png'
        plot_costs(costs=costs_loaded_manually, period=time_between_timesteps,
                   path_to_save=path_to_outer_directory + cost_plot_file)
        costs_loaded_manually = convert_mwh_to_kw_prices(np.array(costs_loaded_manually),
                                                         time_between_timesteps=time_between_timesteps)
        cost_per_hour_man = convert_mwh_to_kw_prices(np.array(cost_per_hour_man),
                                                     time_between_timesteps=time_between_timesteps)




        scheduling_algorithm = LeastLaxityFirstAlg

        beta = 6e3
        charging_days_list = list(evs_timestamp_reset.keys())
        # charging peak is at 360 kw
        power_limit = 400
        power_levels = 10
        eval_env = EVenvironment(scheduling_algorithm=scheduling_algorithm,
                                 time_between_timesteps=time_between_timesteps,
                                 tuning_parameter=beta,
                                 cost_list=costs_loaded_manually,
                                 power_levels=power_levels,
                                 power_limit=150,
                                 charging_days_list=charging_days_list,
                                 train=False,
                                 evse=number_of_evse,
                                 max_charging_rate=maximum_charging_rate)
        day_num = 0
        options = {
            'chosen_day': charging_days_list[day_num],
            'charging_network': charging_networks[1],
            'garages': jpl_garages,
            'document':document
        }

        k_days = len(charging_days_list)
        # possible_gammas = np.arange(0, 1 + 0.1, 0.1)
        possible_gammas = [1]
        max_evs_num_per_day = 0
        for key, value in evs_timestamp_reset.items():
            if len(value) > max_evs_num_per_day:
                max_evs_num_per_day = len(value)
        time_horizon = np.arange(0, int((60 * 24) / time_between_timesteps), 1)
        cumulative_costs_offline_optimal = np.zeros(shape=(len(possible_gammas), k_days))
        offline_optimal_charging_rates = np.zeros(
            shape=(len(possible_gammas), k_days, max_evs_num_per_day, len(time_horizon)))
        number_of_models_tested = 1
        mses_per_env = np.zeros(shape=(number_of_models_tested, k_days))
        mpes_per_env = np.zeros(shape=(number_of_models_tested, k_days))

        model_name = 'sac_1'
        dir_where_saved_models_are = 'SavedModels/ev_experiments1/'
        model = SAC.load(dir_where_saved_models_are + model_name)
        time_horizon = np.arange(0, int((60 * 24) / time_between_timesteps), 1)
        obs, info = eval_env.reset(options=options)
        steps_per_episode = 120
        evaluation_rewards = []
        episode_reward = 0

        for _ in range(steps_per_episode * len(charging_days_list)):

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            # konvertuj kwh na kw v nastaveniach
            if terminated or truncated:
                start_testing_for_cycle = options['chosen_day']
                ch_day = start_testing_for_cycle.day
                ch_mon = start_testing_for_cycle.month
                ch_year = start_testing_for_cycle.year
                path_to_directory = fr'testing_days_jpl/charging_day_{ch_day}_{ch_mon}_{ch_year}/'
                evs_timestamp_reset_list = evs_timestamp_reset[start_testing_for_cycle]
                end_testing_for_cycle = options['chosen_day'] + timedelta(hours=23, minutes=59, seconds=59)
                num_of_evs_given_day = len(evs_timestamp_reset_list)
                for gamma_index, given_gamma in enumerate(possible_gammas, start=0):
                    scheduling_alg = OPT(EVs=evs_timestamp_reset_list,
                                         start=start_testing_for_cycle,
                                         end=end_testing_for_cycle,
                                         gamma=given_gamma,
                                         power_limit=power_limit,
                                         time_between_timesteps=time_between_timesteps,
                                         number_of_evse=number_of_evse,
                                         costs_loaded_manually=costs_loaded_manually)
                    feasibility, opt_charging_rates = scheduling_alg.solve()

                    offline_optimal_charging_rates[gamma_index, day_num, :len(opt_charging_rates),
                    :] = opt_charging_rates
                    cumulative_costs_offline_optimal[gamma_index, day_num] = calculate_cumulative_costs(
                        schedule=opt_charging_rates / (time_between_timesteps / 60),
                        cost_vector=scheduling_alg.cost_vector)



                ut_signals = eval_env.chosen_ut_for_each_timestep
                cumulative_charging_rates = eval_env.chosen_sum_of_charging_rates
                cumulative_charging_rates = np.array(cumulative_charging_rates)
                opt_sum_of_charging_rates = np.sum(offline_optimal_charging_rates[-1, day_num], axis=0)

                eval_env_current_mpe_error = mpe_error_fun_rl_testing(
                    ev_diction=eval_env.delivered_and_undelivered_energy)
                mpes_per_env[0, day_num] = eval_env_current_mpe_error
                eval_env_current_mse_error = mse_error_fun_rl_testing(
                    sum_of_charging_rates=cumulative_charging_rates / (time_between_timesteps / 60),
                    ut_signals=ut_signals,
                    capacity_constraint=power_limit)
                # mses.append(eval_env_current_mse_error)
                mses_per_env[0, day_num] = eval_env_current_mse_error


                comparison_of_algs_file = 'comparison_of_ev_charging_algs.png'
                comparison_of_different_algorithms(
                    cumulative_charging_rates=cumulative_charging_rates / (time_between_timesteps / 60),
                    period=time_between_timesteps,
                    opt_signals=opt_sum_of_charging_rates / (time_between_timesteps / 60),
                    path_to_save=path_to_directory + comparison_of_algs_file)


                pilot_and_real_charging_signal_file = 'comparison_real_pilot.png'
                comparison_pilot_signal_real_signal_graph(ut_signals=ut_signals,
                                                          cumulative_charging_rates=cumulative_charging_rates/(time_between_timesteps/60),
                                                          period=time_between_timesteps,
                                                          path_to_save=path_to_directory + pilot_and_real_charging_signal_file)

                und_energy_file = 'undelivered_energy_offline.txt'
                undelivered_energy_file(filename=path_to_directory + und_energy_file,
                                        evs=evs_timestamp_reset_list[:num_of_evs_given_day],
                                        charging_plan=offline_optimal_charging_rates[-1, day_num])
                und_energy_file = 'undelivered_energy_beta_6000.txt'
                undelivered_energy_file_rl(filename=path_to_directory + und_energy_file,
                                           evs_to_undelivered_dict=eval_env.delivered_and_undelivered_energy)


                evs_data_file = 'evs_data.txt'
                save_evs_to_file(filename=path_to_directory + evs_data_file,
                                 evs_with_time_not_normalised=evs_time_not_normalised[
                                     start_testing_for_cycle],
                                 evs=evs_timestamp_reset_list,
                                 set_maximum_charging_rate=maximum_charging_rate)
                settings_data_file = 'settings.txt'
                create_settings_file(filename=path_to_directory + settings_data_file,
                                     evs_num=len(evs_timestamp_reset_list),
                                     start=start_testing_for_cycle,
                                     end=end_testing_for_cycle,
                                     period=time_between_timesteps,
                                     time_horizon=time_horizon,
                                     cost_function='Ceny elektriny predikovane pre nabuduci den 25.10.2024,(spolocnost CAISO)',
                                     algorithm_name='PPC, Offline optimal',
                                     charging_networks_chosen=charging_networks[0],
                                     garages_chosen=caltech_garages,
                                     operational_constraint=power_limit,
                                     number_of_evse=54,
                                     solver_name='SCIP',
                                     manually_computed_costs_hourly=cost_per_hour_man)
                table_file_name = 'charging_table.csv'
                create_table(charging_profiles_matrix=eval_env.charging_rates_matrix,
                             charging_cost_vector=costs_loaded_manually,
                             period=time_between_timesteps,
                             show_charging_costs=True,
                             path_to_save=path_to_directory + table_file_name,
                             capacity_in_time=ut_signals)

                table_file_name_offline_optimal = 'charging_table_offline_optimal.csv'
                create_table(
                    charging_profiles_matrix=offline_optimal_charging_rates[-1, day_num, :num_of_evs_given_day, :],
                    charging_cost_vector=costs_loaded_manually,
                    period=time_between_timesteps,
                    show_charging_costs=True,
                    path_to_save=path_to_directory + table_file_name_offline_optimal,
                    capacity_in_time=np.sum(offline_optimal_charging_rates[-1, day_num, :num_of_evs_given_day, :],axis=0))
                day_num += 1
                if day_num == len(charging_days_list):
                    break
                options = {
                    'chosen_day': charging_days_list[day_num],
                    'charging_network': charging_networks[1],
                    'garages': jpl_garages,
                    'document':document
                }
                evaluation_rewards.append(episode_reward)
                episode_reward = 0
                obs, info = eval_env.reset(options=options)
        rewards_file = 'evaluation_rewards.txt'
        write_evaluation_rewards_into_file(filename=path_to_outer_directory + rewards_file,
                                           charging_days_list=charging_days_list,
                                           rewards=evaluation_rewards)
        legends_for_each_alg = ['beta = 6000', 'Offline optimal gamma=1']
        color_for_each_alg = ['blue', 'black']
        path_to_outer_directory = fr'testing_days_jpl/'
        costs_per_alg_settings_file = 'costs_depending_on_parameters.png'
        num_of_algs = 1
        mpe_file_per_alg_and_day = 'mpe_per_day_for_different_betas.png'

        mpe_per_day_graph(mpe_values_per_alg=mpes_per_env,
                          legend_names_in_order=[legends_for_each_alg[0]],
                          colors_of_graphs=[color_for_each_alg[0]],
                          path_to_save=path_to_outer_directory + mpe_file_per_alg_and_day)
        mse_file_per_alg_and_day = 'mse_per_day_for_different_betas.png'
        mse_per_day_graph(mse_values_per_alg=mses_per_env,
                          legend_names_in_order=[legends_for_each_alg[0]],
                          colors_of_graphs=[color_for_each_alg[0]],
                          path_to_save=path_to_outer_directory + mse_file_per_alg_and_day)


        cumulative_costs_per_algorithm = np.zeros(shape=(num_of_algs, k_days))
        cumulative_costs_per_algorithm[0] = cumulative_costs_offline_optimal[-1]
        costs_per_day_graph(costs_per_alg=cumulative_costs_per_algorithm,
                            legend_names_in_order=legends_for_each_alg,
                            colors_of_graphs=color_for_each_alg,
                            path_to_save=path_to_outer_directory + costs_per_alg_settings_file)

        mpe_vs_costs_all_days_file = 'mpe_vs_costs_all_days_per_alg.png'
        offline_mpes = [1 - 0.1 * i for i in range(len(possible_gammas))]
        mpe_cost_graph(mpe_values_offline=offline_mpes,
                       mpe_values_env=None,
                       cost_values_offline=np.sum(cumulative_costs_offline_optimal, axis=1),
                       cost_values_env=None,
                       colors_of_graphs=color_for_each_alg,
                       legends_of_graphs=legends_for_each_alg,
                       path_to_save=path_to_outer_directory + mpe_vs_costs_all_days_file)


        # for charging_date in evs_timestamp_reset.keys():
        # draw_barchart_sessions(schedule=[],
        #                        evs_dict_reseted=evs_timestamp_reset)
    a = 0
    def testcaiso2016lmps(self):
        saving_directory = 'caiso_2016/'
        caiso_2016_files = [
            saving_directory + '20160101-20160131 CAISO Day-Ahead Price.csv',
            saving_directory + '20160201-20160229 CAISO Day-Ahead Price.csv',
            saving_directory + '20160301-20160331 CAISO Day-Ahead Price.csv',
            saving_directory + '20160401-20160430 CAISO Day-Ahead Price.csv',
            saving_directory + '20160501-20160531 CAISO Day-Ahead Price.csv',
            saving_directory + '20160601-20160630 CAISO Day-Ahead Price.csv',
            saving_directory + '20160701-20160731 CAISO Day-Ahead Price.csv',
            saving_directory +'20160801-20160831 CAISO Day-Ahead Price.csv',
            saving_directory +'20160901-20160930 CAISO Day-Ahead Price.csv',
            saving_directory +'20161001-20161031 CAISO Day-Ahead Price.csv',
            saving_directory +'20161101-20161130 CAISO Day-Ahead Price.csv',
            saving_directory +'20161201-20161231 CAISO Day-Ahead Price.csv'

        ]
        period = 12
        possible_organisations = ['PGAE', 'SCE', 'SDGE', 'VEA']
        # average_lmps_hourly, average_lmps_per_timestep = load_locational_marginal_prices_per_year(filenames=caiso_2016_files,
        #                                          period=period)
        average_lmps_hourly, average_lmps_per_timestep = load_locational_marginal_prices_per_year(
            filenames=caiso_2016_files,
            period=period)

        a= 0
        plot_costs(costs=average_lmps_per_timestep, period=period)


    def test_correctness_of_acn_static_data(self):
        la_tz = pytz.timezone('America/Los_Angeles')
        # testing phase
        start_testing = la_tz.localize(datetime(2019, 12, 2, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2020, 1, 1, 23, 59, 59))
        maximum_charging_rate = 6.6
        period = 12
        number_of_evse = 54
        number_of_evs_interval = [30, np.inf]
        # this data must be loaded even if environment loads data separately, to filter charging days
        evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
            charging_network=charging_networks[0],
            # garages=caltech_garages,
            garages=caltech_garages,
            start=start_testing,
            end=end_testing,
            period=period,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
            include_overday_charging=False
        )
        plot_hourly_arrivals(evs_time_not_normalised=evs_time_not_normalised_time_reset,
                             title='Arrival times for acn static testing data')
        plot_hourly_departures(evs_time_not_normalised=evs_time_not_normalised_time_reset,
                             title='Departure times for acn static testing data')
        plot_hourly_requested_energy(evs_time_not_normalised=evs_time_not_normalised_time_reset,
                                     title='Energy requested for acn static testing data')
        document = 'acndata_sessions_testing.json'
        evs_timestamp_reset2, evs_timestamp_not_reset2, evs_time_not_normalised2 = get_evs_data_from_document_advanced_settings(
            document=document,
            start=start_testing,
            end=end_testing,
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
            period=period,
            allow_overday_charging=False,
            max_charging_rate_within_interval=[maximum_charging_rate,
                                               maximum_charging_rate],
        )
        plot_hourly_arrivals(evs_time_not_normalised=evs_time_not_normalised2,
                             title='Arrival times for acn testing data')
        plot_hourly_departures(evs_time_not_normalised=evs_time_not_normalised2,
                               title='Departure times for acn  testing data')
        plot_hourly_requested_energy(evs_time_not_normalised=evs_time_not_normalised2,
                                     title='Energy requested for acn testing data')

        start_testing = la_tz.localize(datetime(2018, 11, 1, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2019, 12, 1, 23, 59, 59))

        evs_timestamp_reset3, evs_timestamp_not_reset3, evs_time_not_normalised_time_reset3 = load_time_series_ev_data(
            charging_network=charging_networks[0],
            # garages=caltech_garages,
            garages=caltech_garages,
            start=start_testing,
            end=end_testing,
            period=period,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
            include_overday_charging=False
        )
        plot_hourly_arrivals(evs_time_not_normalised=evs_time_not_normalised_time_reset3,
                             title='Arrival times for acn static training data')
        plot_hourly_departures(evs_time_not_normalised=evs_time_not_normalised_time_reset3,
                               title='Departure times for acn static training data')

        plot_hourly_requested_energy(evs_time_not_normalised=evs_time_not_normalised_time_reset3,
                                     title='Energy requested for acn static training data')
        document1 = 'acndata_sessions_jpl.json'
        start_testing = la_tz.localize(datetime(2019, 12, 2, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2020, 1, 1, 23, 59, 59))
        evs_timestamp_reset_jpl, evs_timestamp_not_reset_jpl, evs_time_not_normalised_jpl = get_evs_data_from_document_advanced_settings(
            document=document1,
            start=start_testing,
            end=end_testing,
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
            period=period,
            allow_overday_charging=False,
            max_charging_rate_within_interval=[maximum_charging_rate,
                                               maximum_charging_rate],
        )
        plot_hourly_arrivals(evs_time_not_normalised=evs_time_not_normalised_jpl,
                             title='Arrival times for acn testing data')
        plot_hourly_departures(evs_time_not_normalised=evs_time_not_normalised_jpl,
                               title='Departure times for acn  testing data')
        plot_hourly_requested_energy(evs_time_not_normalised=evs_time_not_normalised_jpl,
                                     title='Energy requested for acn testing data')

    def test_is_feasible_testing(self):
        la_tz = pytz.timezone('America/Los_Angeles')
        # testing phase
        start_testing = la_tz.localize(datetime(2019, 12, 2, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2020, 1, 1, 23, 59, 59))
        maximum_charging_rate = 3
        period = 12
        number_of_evse = 54
        number_of_evs_interval = [30, np.inf]
        # this data must be loaded even if environment loads data separately, to filter charging days
        evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
            charging_network=charging_networks[0],
            # garages=caltech_garages,
            garages=caltech_garages,
            start=start_testing,
            end=end_testing,
            period=period,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
            include_overday_charging=False
        )
        cost_file = '20241025-20241025 CAISO Day-Ahead Price (1).csv'
        # costs_loaded_manually = load_locational_marginal_prices(filename=cost_file, organization='VEA',period=period)
        costs_loaded_manually = load_locational_marginal_prices(filename=cost_file, organization=None, period=period)
        for charging_date, evs in  evs_timestamp_reset.items():
            end_charging_date = charging_date + timedelta(hours=23,minutes=59, seconds=59)
            scheduling_alg = OPT(EVs=evs,
                                 start=charging_date,
                                 end=end_charging_date,
                                 available_energy_for_each_timestep=0,
                                 ut_interval=[0, 150],
                                 time_between_timesteps=period,
                                 number_of_evse=number_of_evse,
                                 costs_loaded_manually=costs_loaded_manually)
            feasibility, opt_charging_rates = scheduling_alg.solve()
            self.assertTrue(feasibility)
    def test_caltech_trained_model_on_jpl(self):
        la_tz = pytz.timezone('America/Los_Angeles')
        # testing phase
        start_testing = la_tz.localize(datetime(2019, 12, 2, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2020, 1, 1, 23, 59, 59))
        maximum_charging_rate = 6.5
        period = 12
        number_of_evse = 54
        number_of_evs_interval = [30, np.inf]
        document = 'acndata_sessions_jpl.json'
        # this data must be loaded even if environment loads data separately, to filter charging days
        # must match with env settings (reset)
        evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = get_evs_data_from_document_advanced_settings(
            document=document,
            start=start_testing,
            end=end_testing,
            period=period,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
            allow_overday_charging=False
        )
        a = 0



    # this test will return all necessary graphs
    # for easier execution each algorithm here will be executed from scratch for each day
    def test_run_model_on_testing_env(self):
        la_tz = pytz.timezone('America/Los_Angeles')
        # testing phase
        start_testing = la_tz.localize(datetime(2019, 12, 2, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2020, 1, 1, 23, 59, 59))
        # for one timestep
        power_limit = 150
        time_between_timesteps = 12
        maximum_charging_rate = 7
        number_of_evse = 54
        number_of_evs_interval = [30, np.inf]
        document = 'acndata_sessions_testing.json'
        # this data must be loaded even if environment loads data separately, to filter charging days
        # must match with env settings (reset)
        evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
            charging_network=charging_networks[0],
            # garages=caltech_garages,
            garages=caltech_garages,
            start=start_testing,
            end=end_testing,
            period=time_between_timesteps,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
            include_overday_charging=False
        )

        n_cars = np.inf

        # check if badly loaded data are common occurence or it is just situational
        cost_file = '20241025-20241025 CAISO Day-Ahead Price (1).csv'
        # costs_loaded_manually = load_locational_marginal_prices(filename=cost_file, organization='VEA',period=period)
        cost_per_hour_man, costs_loaded_manually = load_locational_marginal_prices(filename=cost_file, organization=None, period=time_between_timesteps)

        path_to_outer_directory = fr'testing_days_caltech/'
        cost_plot_file = 'cost_function.png'
        plot_costs(costs=costs_loaded_manually,
                   period=time_between_timesteps,
                   path_to_save=path_to_outer_directory + cost_plot_file)
        costs_loaded_manually = convert_mwh_to_kw_prices(np.array(costs_loaded_manually), time_between_timesteps=time_between_timesteps)
        cost_per_hour_man = convert_mwh_to_kw_prices(np.array(cost_per_hour_man),time_between_timesteps=time_between_timesteps)


        charging_days_list = list(evs_timestamp_reset.keys())
        k_days = len(charging_days_list)

        number_of_models_tested = 1
        max_evs_num_per_day = 0
        for key, value in evs_timestamp_reset.items():
            if len(value) > max_evs_num_per_day:
                max_evs_num_per_day = len(value)
        legends_for_each_alg = ['beta = 6000', 'Offline optimal gamma=1']
        color_for_each_alg = ['blue', 'black']

        time_horizon = np.arange(0, int((60 * 24) / time_between_timesteps), 1)
        # possible_gammas = np.arange(0, 1 + 0.1, 0.1)
        possible_gammas = [1]
        # possible_betas = [1e3,2e3,4e3,6e3,8e3,1e6]
        possible_betas = [6e3]
        cumulative_costs_offline_optimal = np.zeros(shape=(len(possible_gammas), k_days))
        offline_optimal_charging_rates = np.zeros(
            shape=(len(possible_gammas), k_days, max_evs_num_per_day, len(time_horizon)))
        beta_env_charging_rates = np.zeros(shape=(len(possible_betas), k_days, max_evs_num_per_day, len(time_horizon)))
        mses_per_env = np.zeros(shape=(number_of_models_tested, k_days))
        mpes_per_env = np.zeros(shape=(number_of_models_tested, k_days))
        cumulative_costs_per_env = np.zeros(shape=(1 + number_of_models_tested, k_days))

        num_of_algs = 2
        cumulative_costs_per_algorithm = np.zeros(shape=(num_of_algs, k_days))
        offline_mpes = [1 - 0.1 * i for i in range(len(possible_gammas))]

        mpes_per_env_for_all_days = np.zeros(shape=(number_of_models_tested,))
        mpes_per_alg = np.zeros(shape=(number_of_models_tested,))

        scheduling_algorithm = LeastLaxityFirstAlg
        beta = 6e3
        power_levels = 10
        eval_env = EVenvironment(scheduling_algorithm=scheduling_algorithm,
                            time_between_timesteps=time_between_timesteps,
                            tuning_parameter=beta,
                            cost_list=costs_loaded_manually,
                            power_levels=power_levels,
                            power_limit=power_limit,
                            charging_days_list=charging_days_list,
                            train=False,
                            evse=number_of_evse,
                            max_charging_rate=maximum_charging_rate)

        model_name = 'sac_1'
        dir_where_saved_models_are = 'SavedModels/ev_experiments1/'
        model = SAC.load(dir_where_saved_models_are + model_name)
        day_num = 0
        options = {
            'chosen_day':charging_days_list[day_num]
        }
        steps_per_episode = int((60 * 24) / time_between_timesteps)
        obs, info = eval_env.reset(options=options)
        evaluation_rewards = []
        episode_reward = 0
        for _ in range(steps_per_episode*k_days):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward

            if terminated or truncated:
                start_testing_for_cycle = options['chosen_day']
                evs_timestamp_reset_list = evs_timestamp_reset[start_testing_for_cycle]
                end_testing_for_cycle = options['chosen_day'] + timedelta(hours=23, minutes=59, seconds=59)
                day = options['chosen_day'].day
                month = options['chosen_day'].month
                year = options['chosen_day'].year
                path_to_directory = fr'testing_days_caltech/charging_day_{day}_{month}_{year}/'
                num_of_evs_given_day = len(evs_timestamp_reset_list)
                # num_of_evs_given_day = np.inf
                for gamma_index,given_gamma in enumerate(possible_gammas,start=0):
                    scheduling_alg = OPT(EVs=evs_timestamp_reset_list[:num_of_evs_given_day],
                                         start=start_testing_for_cycle,
                                         end=end_testing_for_cycle,
                                         gamma=given_gamma,
                                         power_limit=power_limit,
                                         time_between_timesteps=time_between_timesteps,
                                         number_of_evse=number_of_evse,
                                         costs_loaded_manually=costs_loaded_manually)
                    feasibility, opt_charging_rates = scheduling_alg.solve()

                    self.assertTrue(feasibility)
                    offline_optimal_charging_rates[gamma_index, day_num,:len(opt_charging_rates),:] = opt_charging_rates

                    cumulative_costs_offline_optimal[gamma_index,day_num] = calculate_cumulative_costs(schedule=opt_charging_rates/(time_between_timesteps/60) ,

                                                                                        cost_vector=scheduling_alg.cost_vector)

                    # offline_optimal_folder = 'offline_optimal/'
                    # table_file_name_offline_optimal = fr'charging_table_offline_optimal_gamma={given_gamma}.csv'
                    # create_table(
                    #     charging_profiles_matrix=offline_optimal_charging_rates[gamma_index, day_num, :num_of_evs_given_day, :],
                    #     charging_cost_vector=costs_loaded_manually,
                    #     period=time_between_timesteps,
                    #     show_charging_costs=True,
                    #     path_to_save=path_to_directory + offline_optimal_folder + table_file_name_offline_optimal)
                ut_signals = np.array(eval_env.chosen_ut_for_each_timestep)
                cumulative_charging_rates = np.array(eval_env.chosen_sum_of_charging_rates)
                cumulative_costs_per_env[0, day_num] = calculate_cumulative_costs_given_ut(uts=ut_signals,
                                                                                           cost_vector=costs_loaded_manually,
                                                                                           period=time_between_timesteps)

                opt_sum_of_charging_rates = np.sum(offline_optimal_charging_rates[-1, day_num], axis=0)

                mpes_per_env[0, day_num] = mpe_error_fun_rl_testing(ev_diction=eval_env.delivered_and_undelivered_energy)
                mses_per_env[0,day_num] = mse_error_fun_rl_testing(sum_of_charging_rates=cumulative_charging_rates/(time_between_timesteps/60),
                                                             ut_signals=ut_signals,
                                                             capacity_constraint=power_limit)
                beta_env_charging_rates[0, day_num, :len(eval_env.charging_rates_matrix),
                :] = eval_env.charging_rates_matrix
                pilot_and_real_charging_signal_file = 'comparison_real_pilot.png'
                comparison_pilot_signal_real_signal_graph(ut_signals=ut_signals,
                                                          cumulative_charging_rates=cumulative_charging_rates/(time_between_timesteps/60),
                                                          period=time_between_timesteps,
                                                          path_to_save=path_to_directory + pilot_and_real_charging_signal_file)




                comparison_of_algs_file = 'comparison_of_ev_charging_algs.png'
                comparison_of_different_algorithms(cumulative_charging_rates=cumulative_charging_rates/(time_between_timesteps/60),
                                                   period=time_between_timesteps,
                                                   opt_signals=opt_sum_of_charging_rates/(time_between_timesteps/60),
                                                   path_to_save=path_to_directory + comparison_of_algs_file)

                und_energy_file = 'undelivered_energy_offline.txt'
                undelivered_energy_file(filename=path_to_directory + und_energy_file,
                                        evs=evs_timestamp_reset_list[:num_of_evs_given_day],
                                        charging_plan=offline_optimal_charging_rates[-1, day_num])
                und_energy_file = 'undelivered_energy_beta_6000.txt'
                undelivered_energy_file_rl(filename=path_to_directory + und_energy_file,
                                           evs_to_undelivered_dict=eval_env.delivered_and_undelivered_energy)


                evs_data_file = 'evs_data.txt'
                save_evs_to_file(filename=path_to_directory + evs_data_file,
                                 evs_with_time_not_normalised=evs_time_not_normalised_time_reset[start_testing_for_cycle],
                                 evs=evs_timestamp_reset_list)
                settings_data_file = 'settings.txt'
                create_settings_file(filename=path_to_directory + settings_data_file,
                                     evs_num=len(evs_timestamp_reset_list),
                                     start=start_testing_for_cycle,
                                     end=end_testing_for_cycle,
                                     period=time_between_timesteps,
                                     time_horizon=time_horizon,
                                     cost_function='Ceny elektriny predikovane pre nabuduci den 25.10.2024,(spolocnost CAISO)',
                                     algorithm_name='PPC, Offline optimal',
                                     charging_networks_chosen=charging_networks[0],
                                     garages_chosen=caltech_garages,
                                     number_of_evse=54,
                                     solver_name='SCIP',
                                     manually_computed_costs_hourly=cost_per_hour_man)

                table_file_name = 'charging_table.csv'
                create_table(charging_profiles_matrix=eval_env.charging_rates_matrix,
                             charging_cost_vector=costs_loaded_manually,
                             capacity_in_time=ut_signals* (time_between_timesteps/60),
                             period=time_between_timesteps,
                             show_charging_costs=True,
                             path_to_save=path_to_directory + table_file_name)

                table_file_name_offline_optimal = 'charging_table_offline_optimal.csv'
                # uts = get_hourly_ut(np.sum(offline_optimal_charging_rates[-1, day_num, :num_of_evs_given_day, :],axis=1),period=time_between_timesteps)
                create_table(charging_profiles_matrix=offline_optimal_charging_rates[-1, day_num,:num_of_evs_given_day,:],
                             charging_cost_vector=costs_loaded_manually,
                             capacity_in_time=np.sum(offline_optimal_charging_rates[-1, day_num, :num_of_evs_given_day, :],axis=0),
                             period=time_between_timesteps,
                             show_charging_costs=True,
                             path_to_save=path_to_directory + table_file_name_offline_optimal)

                print("Reward:", episode_reward)
                evaluation_rewards.append(episode_reward)
                episode_reward = 0.0
                day_num += 1
                if day_num == k_days:
                    break
                options['chosen_day'] = charging_days_list[day_num]
                obs, info = eval_env.reset(options=options)

        rewards_file = 'evaluation_rewards.txt'
        write_evaluation_rewards_into_file(filename=path_to_outer_directory + rewards_file,
                                           charging_days_list=charging_days_list,
                                           rewards=evaluation_rewards)
        mpe_file_per_alg_and_day = 'mpe_per_day_for_different_betas.png'
        mpe_per_day_graph(mpe_values_per_alg=mpes_per_env,
                          legend_names_in_order=[legends_for_each_alg[0]],
                          colors_of_graphs=[color_for_each_alg[0]],
                          path_to_save=path_to_outer_directory + mpe_file_per_alg_and_day)
        mse_file_per_alg_and_day = 'mse_per_day_for_different_betas.png'
        mse_per_day_graph(mse_values_per_alg=mses_per_env,
                          legend_names_in_order=[legends_for_each_alg[0]],
                          colors_of_graphs=[color_for_each_alg[0]],
                          path_to_save=path_to_outer_directory + mse_file_per_alg_and_day)
        cumulative_costs_per_algorithm[1] = deepcopy(cumulative_costs_offline_optimal[-1])
        cumulative_costs_per_algorithm[0] = deepcopy(cumulative_costs_per_env[0])
        costs_per_alg_settings_file = 'costs_depending_on_parameters.png'
        costs_per_day_graph(costs_per_alg=cumulative_costs_per_algorithm,
                            legend_names_in_order=legends_for_each_alg,
                            colors_of_graphs=color_for_each_alg,
                            path_to_save=path_to_outer_directory + costs_per_alg_settings_file)
        # finish first what advisor suggested and then add this
        mpe_vs_costs_all_days_file = 'mpe_vs_costs_all_days_per_alg.png'
        env_mpes = [calculate_mpe_from_charging_rates_over_all_days(charging_rates=beta_env_charging_rates[0],
                                                                                                    evs_for_each_day=evs_timestamp_reset,
                                                                                                    charging_days_list=charging_days_list),1]
        env_costs = [np.sum(cumulative_costs_per_env,axis=1)]*len(offline_mpes)
        mpe_cost_graph(mpe_values_offline=offline_mpes,
                       mpe_values_env=env_mpes,
                       cost_values_offline=np.sum(cumulative_costs_offline_optimal,axis=1),
                       cost_values_env=np.sum(cumulative_costs_per_env,axis=1) + [0],
                       colors_of_graphs=color_for_each_alg,
                       legends_of_graphs=legends_for_each_alg,
                       path_to_save=path_to_outer_directory + mpe_vs_costs_all_days_file)
    # TODO: currently doing this test
    def testRLourenvlibrary(self):
        la_tz = pytz.timezone('America/Los_Angeles')
        start_testing = la_tz.localize(datetime(2018, 11, 1, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2019, 12, 1, 23, 59, 59))
        # maximum_charging_rate = 6.6
        maximum_charging_rate = 7
        period = 12
        max_number_of_episodes = 500
        number_of_timesteps_in_one_episode = ((60 * 24) / period)
        total_timesteps = max_number_of_episodes * number_of_timesteps_in_one_episode
        number_of_evs_interval = [30, np.inf]
        # this data must be loaded even if environment loads data separately, to filter charging days
        evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
            charging_network=charging_networks[0],
            # garages=caltech_garages,
            garages=caltech_garages,
            start=start_testing,
            end=end_testing,
            period=period,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
            include_overday_charging=False

        )
        charging_days_list = list(evs_timestamp_reset.keys())
        # TODO: have forgotten to save the model
        # make environment faster
        # # haarnoja - even simpler tasks sometimes need 1000 000 steps to learn
        # total_timesteps = int(500000 / 5)
        total_timesteps = 100000
        # total_timesteps = 240000
        scheduling_algorithm = LeastLaxityFirstAlg
        beta = 6e3
        # check if badly loaded data are common occurence or it is just situational
        # cost_file = '20241025-20241025 CAISO Day-Ahead Price (1).csv'
        # costs_loaded_manually = load_locational_marginal_prices(filename=cost_file, organization='VEA',period=period)
        # costs_loaded_manually = load_locational_marginal_prices(filename=cost_file, organization=None, period=period)
        # different cost function
        ts = np.arange(0, 24, (period/60))
        costs_loaded_manually = [(1-(ts[i]/24)) for i in range(len(ts))]
        power_limit = 150
        power_levels = 10
        env = EVenvironment(scheduling_algorithm=scheduling_algorithm,
                            time_between_timesteps=period,
                            tuning_parameter=beta,
                            max_charging_rate=maximum_charging_rate,
                            cost_list=costs_loaded_manually,
                            power_levels=power_levels,
                            power_limit=power_limit,
                            charging_days_list=charging_days_list,
                            limit_ramp_rates=True)



        model = SAC("MlpPolicy",
                    env,
                    # gamma=0.5,
                    gamma=0.5,
                    ent_coef=0.5,
                    tensorboard_log="./sac/",
                    verbose=1)
        # reward_function_form = ''
        model.learn(total_timesteps=total_timesteps, callback=TensorboardCallback())

        model_name = 'sac_1'
        dir_where_saved_models_are = 'SavedModels/ev_experiments1/'
        model.save(dir_where_saved_models_are+model_name)
        # hyp_diction = {
        #     'o3':0.6,
        #     'gamma':0.99,
        #                'ramp limits':True}
        # save_to_json(data=hyp_diction,
        #              filename=dir_where_saved_models_are +model_name +'.json')

    def test_load_saved_model(self):
        ...

    def test_res_graph(self):
        la_tz = pytz.timezone('America/Los_Angeles')
        start_testing = la_tz.localize(datetime(2019, 12, 16, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2019, 12, 16, 23, 59, 59))

        ut_interval = [0, 150]
        maximum_charging_rate = 6.6
        number_of_evse = 54
        period = 12
        number_of_evs_interval = [30, np.inf]
        evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
            charging_network=charging_networks[0],
            # garages=caltech_garages,
            garages=caltech_garages[:1],
            start=start_testing,
            end=end_testing,
            period=period,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
            include_overday_charging=False
        )

        evs_timestamp_not_reset_list = convert_evs_diction_to_array(evs_diction=evs_timestamp_not_reset)



        cost_file = '20241025-20241025 CAISO Day-Ahead Price (1).csv'
            # TODO: pozriet sa na data lebo sa zda ze cerpam zo zlych dat
        a = 0
        # costs_loaded_manually = load_locational_marginal_prices(filename=cost_file, organization='VEA',period=period)
        costs_loaded_manually = load_locational_marginal_prices(filename=cost_file, organization=None, period=period)
        plot_costs(costs=costs_loaded_manually, period=period)
        available_energy_for_each_timestep = 1000
        scheduling_alg = OPT(EVs=evs_timestamp_not_reset_list,
                             start=start_testing,
                             end=end_testing,
                             available_energy_for_each_timestep=available_energy_for_each_timestep,
                             ut_interval=ut_interval,
                             time_between_timesteps=period,
                             number_of_evse=number_of_evse,
                             costs_loaded_manually=costs_loaded_manually
                             # cost_function=cost_function

                             )
        feasibility, charging_rates = scheduling_alg.solve()
        cost_vector = scheduling_alg.cost_vector

        # cost_vector = scheduling_alg.get_cost_vector()

        charging_rates_within_bounds = check_charging_rates_within_bounds(evs=evs_timestamp_not_reset_list,
                                                                          charging_rates=charging_rates,
                                                                          )
        self.assertTrue(charging_rates_within_bounds)

        infrastructure_not_violated = check_infrastructure_not_violated(charging_rates=charging_rates,
                                                                        available_energy_for_each_timestep=scheduling_alg.available_energy_for_each_timestep)

        self.assertTrue(infrastructure_not_violated)

        all_energy_demands_met = check_all_energy_demands_met(evs=evs_timestamp_not_reset_list,
                                                              charging_rates=charging_rates,
                                                              algorithm_name=scheduling_alg.algorithm_name)
        self.assertTrue(all_energy_demands_met)

        draw_barchart_sessions(schedule=charging_rates, evs_dict_reseted=evs_timestamp_reset)
        mpe_values = []
        costs = []

        given_day = start_testing
        i = 0
        while given_day <= end_testing:
            timesteps_of_one_day = int((24*60)/period)
            start_of_day_timesteps = i*timesteps_of_one_day
            end_of_day_timesteps = (i + 1)*timesteps_of_one_day
            schedule_for_given_day = charging_rates[:,start_of_day_timesteps:end_of_day_timesteps]
            # evs_for_given_day = evs_timestamp_reset[given_day]
            evs_for_given_day = evs_timestamp_not_reset_list

            costs_for_given_day = calculate_cumulative_costs(schedule=schedule_for_given_day,
                                                             cost_vector=scheduling_alg.cost_vector)

            charging_in_time_graph(ut_signals_offline=get_ut_signals_from_schedule(schedule=schedule_for_given_day),period=period)
            mpe_values.append(mpe_error_fun(schedule=schedule_for_given_day, evs=evs_for_given_day))
            costs.append(costs_for_given_day)
            given_day += timedelta(days=1)
            i += 1
            # mpe = mpe_error_fun()
            # mpe_values.append()
        mpe_per_day_graph(mpe_values=mpe_values)
        costs_per_day_graph(costs=costs)
        mpe_cost_graph(mpe_values=mpe_values, cost_values=costs)
    # more will be added in future
    #      since the switch to paid charging 1.Nov 2018 onwards, ACN had less users
    #  for now tested on one day
    # lets say it looks a bit similar but this is only for one day
    def test_res_graphs_for_k_days(self):
        la_tz = pytz.timezone('America/Los_Angeles')
        # start_testing = la_tz.localize(datetime(2019, 12, 2, 0, 0, 0))
        # end_testing = la_tz.localize (datetime(2020, 1, 1, 23, 59, 59))
        start_testing = la_tz.localize(datetime(2019, 12, 16, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2019, 12, 16, 23, 59, 59))
        ut_interval = [0, 150]
        maximum_charging_rate = 6.6
        number_of_evse = 54
        period = 12
        number_of_evs_interval = [30, np.inf]
        evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
            charging_network=charging_networks[0],
            # garages=caltech_garages,
            garages=caltech_garages[:1],
            start=start_testing,
            end=end_testing,
            period=period,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
        )

        evs_timestamp_not_reset_list = convert_evs_diction_to_array(evs_diction=evs_timestamp_not_reset)

        gamma = np.arange(0,1.1,0.1)

        cost_file = '20241025-20241025 CAISO Day-Ahead Price (1).csv'
        # costs_loaded_manually = load_locational_marginal_prices(filename=cost_file, organization='VEA',period=period)
        costs_loaded_manually = load_locational_marginal_prices(filename=cost_file, organization=None, period=period)
        plot_costs(costs=costs_loaded_manually, period=period)
        available_energy_for_each_timestep = 1000
        mpe_values = np.zeros(shape=(len(gamma)))
        cost_values = np.zeros(shape=(len(gamma)))
        index_of_values = len(gamma) - 1
        # cost_values = np.zeros(shape=(len(gamma))) + math.fsum(costs_loaded_manually)
        # works with gamma
        z = 0

        # is u discrete check if it works with it
        for i, chosen_gamma in enumerate(gamma, start=0):
            scheduling_alg = OPT(EVs=evs_timestamp_not_reset_list,
                                 start=start_testing,
                                 end=end_testing,
                                 available_energy_for_each_timestep=available_energy_for_each_timestep,
                                 ut_interval=ut_interval,
                                 time_between_timesteps=period,
                                 number_of_evse=number_of_evse,
                                 costs_loaded_manually=costs_loaded_manually,
                                 gamma=chosen_gamma,
                                 # cost_function=cost_function

                                 # doesnt work for tested for discrete u
                                 # is_u_discrete=True
                                 )
            feasibility, charging_rates = scheduling_alg.solve(verbose=True)
            cost_vector = scheduling_alg.cost_vector
            # compute costs from schedule
            mpe_values[i] = mpe_error_fun(schedule=charging_rates, evs=evs_timestamp_not_reset_list)
            cost_values[i] = calculate_cumulative_costs(schedule=charging_rates, cost_vector=cost_vector)
            # cost_vector = scheduling_alg.get_cost_vector()

            charging_rates_within_bounds = check_charging_rates_within_bounds(evs=evs_timestamp_not_reset_list,
                                                                              charging_rates=charging_rates,
                                                                              )
            self.assertTrue(charging_rates_within_bounds)

            infrastructure_not_violated = check_infrastructure_not_violated(charging_rates=charging_rates,
                                                                            available_energy_for_each_timestep=scheduling_alg.available_energy_for_each_timestep)

            self.assertTrue(infrastructure_not_violated)
            all_energy_demands_met = check_all_energy_demands_met(evs=evs_timestamp_not_reset_list,
                                                                  charging_rates=charging_rates,
                                                                  gamma=chosen_gamma,
                                                                  algorithm_name=scheduling_alg.algorithm_name)
            self.assertTrue(all_energy_demands_met)
        mpe_cost_graph(mpe_values=mpe_values, cost_values=cost_values)
        # draw_barchart_sessions(schedule=charging_rates, evs_dict_reseted=evs_timestamp_reset)
        # mpe_values = []
        # costs = []
        #
        # given_day = start_testing
        # i = 0
        # while given_day <= end_testing:
        #     timesteps_of_one_day = int((24*60)/period)
        #     start_of_day_timesteps = i*timesteps_of_one_day
        #     end_of_day_timesteps = (i + 1)*timesteps_of_one_day
        #     schedule_for_given_day = charging_rates[:,start_of_day_timesteps:end_of_day_timesteps]
        #     # evs_for_given_day = evs_timestamp_reset[given_day]
        #     evs_for_given_day = evs_timestamp_not_reset_list
        #
        #     costs_for_given_day = calculate_cumulative_costs(schedule=schedule_for_given_day,
        #                                                      cost_vector=scheduling_alg.cost_vector)
        #     # in case of offline optimal we dont check for now the real charging power vs substation charging power they should be the same
        #     charging_in_time_graph(ut_signals_offline=get_ut_signals_from_schedule(schedule=schedule_for_given_day),period=period)
        #     mpe_values.append(mpe_error_fun(schedule=schedule_for_given_day, evs=evs_for_given_day))
        #     costs.append(costs_for_given_day)
        #     given_day += timedelta(days=1)
        #     i += 1
        #     # mpe = mpe_error_fun()
        #     # mpe_values.append()
        #
        #
        # overall_mpe_values_for_all_k_days = mpe_error_fun(schedule=charging_rates, evs=evs_timestamp_not_reset_list)
        # mpe_per_day_graph(mpe_values=mpe_values)
        # costs_per_day_graph(costs=costs)



    def test1(self):
        filename = 'acndata_sessions_acn.json'
        txt_ev_info_filename = 'evs_data.txt'
        settings_filename = 'settings.txt'
        start_testing = datetime(2018, 4, 26, 0, 0, 0)
        end_testing = datetime(2018, 4, 26, 23, 59, 59)

        ut_interval = [0, 150]
        maximum_charging_rate = 6.6
        number_of_evse = 54
        period = 12
        number_of_evs_interval = [0, np.inf]
        evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
            charging_network=charging_networks[0],
            # garages=caltech_garages,
            garages=caltech_garages[:1],
            start=start_testing,
            end=end_testing,
            period=period,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=True,

        )
        # 1 badly loaded file under such conditions
        a = 0
        # # seems OK
        # this tests is for debugging purposes to show that there has been a decrease
        # in number of sessions since Sep 1. 2018- 1. Nov 2019
    def test1_for_more_charging_trend_before(self):

        la_tz = pytz.timezone('America/Los_Angeles')
        start_testing = la_tz.localize(datetime(2018, 9, 1, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2018, 11, 1, 23, 59, 59))

        # end_testing = datetime(2019, 12, 10, 23, 59, 59, tzinfo=timezone.utc)
        ut_interval = [0, 150]
        maximum_charging_rate = 6.6
        number_of_evse = 54
        period = 12
        number_of_evs_interval = [0, np.inf]
        evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
            charging_network=charging_networks[0],
            # garages=caltech_garages,
            garages=caltech_garages,
            start=start_testing,
            end=end_testing,
            period=period,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=True,

        )

        # 1 badly loaded file under such conditions
        a = 0

    # OK results
    # # seems ok as well matching with acndata article
    # this tests is for debugging purposes to show that there has been a decrease
    # in number of sessions since Nov 1. 2018- 1. Jan 2019
    def test1_for_less_charging_trend(self):
        charging_networks = ['caltech', 'jpl', 'office_01']
        # find which exact garage we will use

        # 3 days should have at least 60 charging sessions out of 14
        # possibly we need to choose all garages
        caltech_garages = ['California_Garage_01',
                           'California_Garage_02',
                           'LIGO_01',
                           'N_Wilson_Garage_01',
                           'S_Wilson_Garage_01']
        jpl_garages = ['Arroyo_Garage_01']
        office_01_garages = ['Parking_Lot_01']

        # start_testing = datetime(2019, 12, 2, 0, 0, 0,tzinfo=timezone.utc)
        # end_testing = datetime(2020, 1, 1, 23, 59, 59,  tzinfo=timezone.utc)

        # Define the Los Angeles timezone
        la_tz = pytz.timezone('America/Los_Angeles')
        start_testing = la_tz.localize(datetime(2018, 11, 1, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2019, 1, 1, 23, 59, 59))

        # end_testing = datetime(2019, 12, 10, 23, 59, 59, tzinfo=timezone.utc)
        ut_interval = [0, 150]
        maximum_charging_rate = 6.6
        number_of_evse = 54
        period = 12
        number_of_evs_interval = [0, np.inf]
        evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
            charging_network=charging_networks[0],
            # garages=caltech_garages,
            garages=caltech_garages[:1],
            start=start_testing,
            end=end_testing,
            period=period,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=True,

        )
        a = 0


    def test1_for_k_test_days(self):
        charging_networks = ['caltech', 'jpl', 'office_01']
        # find which exact garage we will use

        # 3 days should have at least 60 charging sessions out of 14
        # possibly we need to choose all garages
        caltech_garages = ['California_Garage_01',
                           'California_Garage_02',
                           'LIGO_01',
                           'N_Wilson_Garage_01',
                           'S_Wilson_Garage_01']
        jpl_garages = ['Arroyo_Garage_01']
        office_01_garages = ['Parking_Lot_01']

        # start_testing = datetime(2019, 12, 2, 0, 0, 0,tzinfo=timezone.utc)
        # end_testing = datetime(2020, 1, 1, 23, 59, 59,  tzinfo=timezone.utc)

        # Define the Los Angeles timezone
        la_tz = pytz.timezone('America/Los_Angeles')

        # start_testing = datetime(2019, 11, 2, 0, 0, 0, tzinfo=timezone.utc)
        # end_testing = datetime(2020, 1, 1, 23, 59, 59, tzinfo=timezone.utc)

        start_testing = la_tz.localize(datetime(2019, 12, 2, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2020, 1, 1, 23, 59, 59))

        # end_testing = datetime(2019, 12, 10, 23, 59, 59, tzinfo=timezone.utc)
        ut_interval = [0, 150]
        maximum_charging_rate = 6.6
        number_of_evse = 54
        period = 12
        number_of_evs_interval = [30, np.inf]
        evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
            charging_network=charging_networks[0],
            # garages=caltech_garages,
            garages=caltech_garages[:1],
            start=start_testing,
            end=end_testing,
            period=period,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,

        )
        a = 0
        print('dlzka evs timestamp',len(evs_timestamp_reset))
        print(evs_timestamp_reset.keys())
        # given_day = start_testing
        # while given_day <= end_testing:
        for given_day in list(evs_timestamp_reset.keys()):
            if evs_timestamp_reset.get(given_day, None) is None:
                print('is None')
                given_day += timedelta(days=1)
                continue
            print('is not None')
            evs_for_given_day = evs_timestamp_reset[given_day]
            # plot_arrivals_for_given_day(evs=evs_for_given_day,day=given_day,period=period)
            # plot_departures_for_given_day(evs=evs_for_given_day,day=given_day,period=period)
            draw_barchart_sessions(schedule=[],evs_dict_reseted=evs_timestamp_reset)
            given_day += timedelta(days=1)