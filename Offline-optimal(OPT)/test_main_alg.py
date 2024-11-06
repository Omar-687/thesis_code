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
            garages=caltech_garages[:1],
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
            include_weekends=True,
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
        a = 0
    # this test will return all necessary graphs
    # for easier execution each algorithm here will be executed from scratch for each day
    def test_run_model_on_testing_env(self):
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
            garages=caltech_garages[:1],
            start=start_testing,
            end=end_testing,
            period=period,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
            include_overday_charging=True
        )
        charging_days_list = list(evs_timestamp_reset.keys())
        scheduling_algorithm = LeastLaxityFirstAlg
        beta = 6e3
        # check if badly loaded data are common occurence or it is just situational
        cost_file = '20241025-20241025 CAISO Day-Ahead Price (1).csv'
        # costs_loaded_manually = load_locational_marginal_prices(filename=cost_file, organization='VEA',period=period)
        costs_loaded_manually = load_locational_marginal_prices(filename=cost_file, organization=None, period=period)
        plot_costs(costs=costs_loaded_manually, period=period)
        operator_constraint = 150
        eval_env = EVenvironment(scheduling_algorithm=scheduling_algorithm,
                            time_between_timesteps=period,
                            tuning_parameter=beta,
                            cost_list=costs_loaded_manually,
                            power_levels=10,
                            power_rating=operator_constraint,
                            charging_days_list=charging_days_list,
                            train=False,
                            evse=number_of_evse)
        # evs_timestamp_not_reset_list = convert_evs_diction_to_array(evs_diction=evs_timestamp_not_reset)


        # add solving by opt
        # scheduling_alg
        model_name = 'sac_1'
        dir_where_saved_models_are = 'SavedModels/ev_experiments1/'
        model = SAC.load(dir_where_saved_models_are + model_name)
        i = 0
        options = {
            'chosen_day':charging_days_list[i]
        }
        obs, info = eval_env.reset(options=options)
        k_days = 14
        steps_per_episode = int((60*24)/period)

        legends_for_each_alg = ['beta = 6000','Offline optimal gamma=1']
        color_for_each_alg = ['blue', 'black']
        episode_reward = 0
        mpes = []
        mpes_offline_optimal = []
        cumulative_costs_for_all_algorithms = []
        cumulative_costs = []
        cumulative_costs_offline_optimal = []
        mses = []
        for _ in range(steps_per_episode*k_days):
            if i == k_days:
                break
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward

            if terminated or truncated:
                start_testing_for_cycle = options['chosen_day']
                evs_timestamp_not_reset_list = evs_timestamp_reset[start_testing_for_cycle]
                end_testing_for_cycle = options['chosen_day'] + timedelta(hours=23, minutes=59, seconds=59)
                scheduling_alg = OPT(EVs=evs_timestamp_not_reset_list,
                                     start=start_testing_for_cycle,
                                     end=end_testing_for_cycle,
                                     available_energy_for_each_timestep=0,
                                     ut_interval=[0, 150],
                                     time_between_timesteps=period,
                                     number_of_evse=number_of_evse,
                                     costs_loaded_manually=costs_loaded_manually)
                feasibility, opt_charging_rates = scheduling_alg.solve()
                opt_sum_of_charging_rates = []
                for col in range(opt_charging_rates.shape[1]):
                    opt_sum_of_charging_rates.append(math.fsum(opt_charging_rates[:, col]))



                ut_signals = eval_env.chosen_ut_for_each_timestep
                cumulative_charging_rates = eval_env.chosen_sum_of_charging_rates
                cumulative_costs.append(eval_env.cumulative_costs)
                # add cumulative costs for opt too
                cumulative_charging_costs_for_one_day_offline_optimal = calculate_cumulative_costs(schedule=opt_charging_rates,
                                                                                                   cost_vector=costs_loaded_manually)
                cumulative_costs_offline_optimal.append(cumulative_charging_costs_for_one_day_offline_optimal)

                eval_env_current_mpe_error = mpe_error_fun_rl_testing(ev_diction=eval_env.delivered_and_undelivered_energy)
                mpes.append(eval_env_current_mpe_error)

                eval_env_current_mse_error = mse_error_fun_rl_testing(sum_of_charging_rates=cumulative_charging_rates,
                                                             ut_signals=ut_signals,
                                                             capacity_constraint=operator_constraint)
                mses.append(eval_env_current_mse_error)
                comparison_pilot_signal_real_signal_graph(ut_signals=ut_signals,
                                                          cumulative_charging_rates=cumulative_charging_rates,
                                                          period=period,
                                                          opt_signals=opt_sum_of_charging_rates)

                draw_barchart_sessions_from_RL(dict_of_evs=eval_env.delivered_and_undelivered_energy,
                                               dict_of_arrivals_and_departures=eval_env.dict_arrivals_departures,
                                               charging_date=eval_env.charging_date)
                print("Reward:", episode_reward)
                episode_reward = 0.0
                i += 1
                if i == k_days:
                    break
                options['chosen_day'] = charging_days_list[i]
                obs, info = eval_env.reset(options=options)
        cumulative_costs_offline_optimal = np.array(cumulative_costs_offline_optimal)
        cumulative_costs = np.array(cumulative_costs)
        all_cumulative_costs = []
        # all_cumulative_costs.append(cumulative_costs)
        all_cumulative_costs.append(cumulative_costs_offline_optimal)
        all_cumulative_costs = np.array(all_cumulative_costs)
        mpe_per_day_graph(mpe_values_per_alg=mpes,
                          legend_names_in_order=legends_for_each_alg[1:],
                          colors_of_graphs=color_for_each_alg[1:])
        mse_per_day_graph(mse_values_per_alg=mses,
                          legend_names_in_order=legends_for_each_alg[1:],
                          colors_of_graphs=color_for_each_alg[1:])
        # costs_per_day_graph(costs_per_alg=all_cumulative_costs,
        #                     legend_names_in_order=legends_for_each_alg,
        #                     colors_of_graphs=color_for_each_alg)
        costs_per_day_graph(costs_per_alg=all_cumulative_costs,
                            legend_names_in_order=legends_for_each_alg[1:],
                            colors_of_graphs=color_for_each_alg[1:])
    # TODO: currently doing this test
    def testRLourenvlibrary(self):
        la_tz = pytz.timezone('America/Los_Angeles')
        start_testing = la_tz.localize(datetime(2018, 11, 1, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2019, 12, 1, 23, 59, 59))
        maximum_charging_rate = 6.6
        period = 12
        max_number_of_episodes = 500
        number_of_timesteps_in_one_episode = ((60 * 24) / period)
        total_timesteps = max_number_of_episodes * number_of_timesteps_in_one_episode
        number_of_evs_interval = [30, np.inf]
        # this data must be loaded even if environment loads data separately, to filter charging days
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
        charging_days_list = list(evs_timestamp_reset.keys())
        # TODO: have forgotten to save the model
        # make environment faster
        # # haarnoja - even simpler tasks sometimes need 1000 000 steps to learn
        # total_timesteps = int(500000 / 5)
        total_timesteps = 100000
        scheduling_algorithm = LeastLaxityFirstAlg
        beta = 6e3
        # check if badly loaded data are common occurence or it is just situational
        # cost_file = '20241025-20241025 CAISO Day-Ahead Price (1).csv'
        # costs_loaded_manually = load_locational_marginal_prices(filename=cost_file, organization='VEA',period=period)
        # costs_loaded_manually = load_locational_marginal_prices(filename=cost_file, organization=None, period=period)
        # different cost function
        ts = np.arange(0, 24, (period/60))
        costs_loaded_manually = [(1-(ts[i]/24)) for i in range(len(ts))]
        env = EVenvironment(scheduling_algorithm=scheduling_algorithm,
                            time_between_timesteps=period,
                            tuning_parameter=beta,
                            cost_list=costs_loaded_manually,
                            power_levels=10,
                            power_rating=150,
                            charging_days_list=charging_days_list,
                            limit_ramp_rates=True)



        model = SAC("MlpPolicy",
                    env,
                    gamma=0.5,
                    ent_coef=0.5,
                    tensorboard_log="./sac/",
                    verbose=1)
        reward_function_form = ''
        model.learn(total_timesteps=total_timesteps, callback=TensorboardCallback())

        model_name = 'sac_1'
        dir_where_saved_models_are = 'SavedModels/ev_experiments1/'
        model.save(dir_where_saved_models_are+model_name)
        hyp_diction = {
            'o3':0.6,
            'gamma':0.99,
                       'ramp limits':True}
        save_to_json(data=hyp_diction,
                     filename=dir_where_saved_models_are +model_name +'.json')

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