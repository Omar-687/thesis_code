import numpy as np
import pytz
from datetime import datetime, timedelta
import random
import pandas as pd
import json
import math
from acnportal import acnsim
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



        # TODO:this we test

    # HERE we print results some error here possibly after big changes
    def test_algorithms_on_testing_datasets(self):
        la_tz = pytz.timezone('America/Los_Angeles')
        # testing phase
        start_testing = la_tz.localize(datetime(2019, 12, 2, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2020, 1, 1, 23, 59, 59))
        maximum_charging_rate = 10
        time_between_timesteps = 12
        number_of_evse = 54
        power_limit = 150
        possible_sites = ['caltech', 'jpl']
        site = 'caltech'
        number_of_evs_interval = [30, np.inf]
        document = f'{site}_data_testing_save.json'
        # this data must be loaded even if environment loads data separately, to filter charging days
        evs_timestamp_reset,evs_timestamp_not_reset, evs_time_not_normalised = (
            get_evs_data_from_document_advanced_settings(document=document,
                                                         start=start_testing,
                                                         end=end_testing,
                                                         number_of_evs_interval=number_of_evs_interval,
                                                         include_weekends=False,
                                                         allow_overday_charging=False,
                                                         period=time_between_timesteps,
                                                         max_charging_rate_within_interval=[maximum_charging_rate,maximum_charging_rate],
                                                         dates_in_ios_format=True))
        charging_days_list = list(evs_timestamp_reset.keys())

        cost_file = '20241025-20241025 CAISO Day-Ahead Price (1).csv'
        # costs_loaded_manually = load_locational_marginal_prices(filename=cost_file, organization='VEA',period=period)
        cost_per_hour_man, costs_loaded_manually = load_locational_marginal_prices(filename=cost_file,
                                                                                   organization=None,
                                                                                   period=time_between_timesteps)

        path_to_outer_directory = fr'testing_days_{site}/'
        cost_plot_file = 'cost_function.png'
        plot_costs(costs=costs_loaded_manually, period=time_between_timesteps,
                   path_to_save=path_to_outer_directory + cost_plot_file)
        costs_loaded_manually = convert_mwh_to_kwh_prices(np.array(costs_loaded_manually),
                                                          time_between_timesteps=time_between_timesteps)
        cost_per_hour_man = convert_mwh_to_kwh_prices(np.array(cost_per_hour_man),
                                                      time_between_timesteps=time_between_timesteps)


        time_horizon = np.arange(0, int((60 * 24) / time_between_timesteps), 1)
        for charging_date in charging_days_list:
            ch_day = charging_date.day
            ch_mon = charging_date.month
            ch_year = charging_date.year
            new_folder = fr'testing_days_{site}/charging_day_{ch_day}_{ch_mon}_{ch_year}'
            os.makedirs(new_folder, exist_ok=True)
            path_to_directory = fr'testing_days_{site}/charging_day_{ch_day}_{ch_mon}_{ch_year}/'

            start_testing_for_cycle = charging_date
            evs_timestamp_reset_list = evs_timestamp_reset[start_testing_for_cycle]
            end_testing_for_cycle = start_testing_for_cycle + timedelta(hours=23, minutes=59, seconds=59)
            evs_data_file = 'evs_data.txt'
            save_evs_to_file(filename=path_to_directory + evs_data_file,
                             evs_with_time_not_normalised=evs_time_not_normalised[
                                 charging_date],
                             sort_by='arrival_time',
                             sort_in_order='asc',
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
                                 algorithm_name='PPC (s jednoduchym planovacim algoritmom LLF), Offline optimal',
                                 charging_networks_chosen=site,
                                 garages_chosen=caltech_garages,
                                 operational_constraint=power_limit,
                                 number_of_evse=54,
                                 solver_name='SCIP',
                                 manually_computed_costs_hourly=cost_per_hour_man)



        scheduling_algorithm = LeastLaxityFirstAlg
        # scheduling_algorithm = SmoothedLeastLaxityAlg
        charging_days_list = list(evs_timestamp_reset.keys())

        k_days = len(charging_days_list)
        # possible_gammas = np.arange(0, 1 + 0.1, 0.1)
        possible_gammas = [1]
        max_evs_num_per_day = 0
        for key, value in evs_timestamp_reset.items():
            if len(value) > max_evs_num_per_day:
                max_evs_num_per_day = len(value)
        cumulative_costs_offline_optimal = np.zeros(shape=(len(possible_gammas), k_days))

        offline_optimal_charging_rates = np.zeros(
            shape=(len(possible_gammas), k_days, max_evs_num_per_day, len(time_horizon)))

        for day_num, charging_day in enumerate(charging_days_list, start=0):
            start_testing_for_cycle = charging_day
            ch_day = charging_day.day
            ch_mon = charging_day.month
            ch_year = charging_day.year
            path_to_directory = fr'testing_days_{site}/charging_day_{ch_day}_{ch_mon}_{ch_year}/'
            evs_timestamp_reset_list = evs_timestamp_reset[start_testing_for_cycle]
            num_of_evs_given_day = len(evs_timestamp_reset_list)
            end_testing_for_cycle = start_testing_for_cycle + timedelta(hours=23, minutes=59, seconds=59)
            for gamma_index, given_gamma in enumerate(possible_gammas, start=0):
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
                offline_optimal_charging_rates[gamma_index, day_num, :len(opt_charging_rates), :] = opt_charging_rates

                cumulative_costs_offline_optimal[gamma_index, day_num] = calculate_cumulative_costs(
                    schedule=opt_charging_rates, cost_vector=scheduling_alg.cost_vector)
            table_file_name_offline_optimal = 'charging_table_offline_optimal.csv'
            create_table(
                charging_profiles_matrix=offline_optimal_charging_rates[-1, day_num, :num_of_evs_given_day, :],
                charging_cost_vector=costs_loaded_manually,
                period=time_between_timesteps,
                show_charging_costs=True,
                path_to_save=path_to_directory + table_file_name_offline_optimal,
                capacity_in_time=np.sum(offline_optimal_charging_rates[-1, day_num, :num_of_evs_given_day, :], axis=0))
            und_energy_file = 'undelivered_energy_offline.txt'
            undelivered_energy_file(filename=path_to_directory + und_energy_file,
                                    evs=evs_timestamp_reset_list[:num_of_evs_given_day],
                                    charging_plan=offline_optimal_charging_rates[-1, day_num])

        steps_per_episode = 120
        evaluation_rewards = []
        episode_reward = 0
        possible_betas = [1e6]
        # test models with different beta or same beta (it is also possible, just change legend so you can read properly)
        # manually change beta in string and also in array
        # without zip, it loads automatically the zip file
        # models_directories = [f'resulting_models/beta=1e6/{site}/sac_10',
        #                       f'resulting_models/beta=6e3/{site}/sac_13']
        models_directories = ['SavedModels/ev_experiments1/sac_25']
        # models_directories = ['SavedModels/ev_experiments1/sac_44']
        # models_directories = ['SavedModels/ev_experiments1/sac_46']
        # models_directories = ['SavedModels/ev_experiments1/sac_33'] sllf same param as sac_25
        # models_directories = ['SavedModels/ev_experiments1/sac_35']
        number_of_models_tested = len(models_directories)
        cumulative_costs_per_env = np.zeros(shape=(1 + number_of_models_tested, k_days))
        mses_per_env = np.zeros(shape=(number_of_models_tested, k_days))
        mpes_per_env = np.zeros(shape=(number_of_models_tested, k_days))

        power_levels_per_env = [10]
        power_limits_per_env = [150]
        max_ramp_rates_per_env = [30]
        for model_index, chosen_beta in enumerate(possible_betas, start=0):
            # loading of different setups in future, maybe through json file
            eval_env = EVenvironment(scheduling_algorithm=scheduling_algorithm,
                                     time_between_timesteps=time_between_timesteps,
                                     tuning_parameter=chosen_beta,
                                     cost_list=costs_loaded_manually,
                                     power_levels=power_levels_per_env[model_index],
                                     power_limit=power_limits_per_env[model_index],
                                     train=False,
                                     evse=number_of_evse,
                                     max_charging_rate=maximum_charging_rate,
                                     costs_in_kwh=True,
                                     max_ramp_rate=max_ramp_rates_per_env[model_index])
            model = SAC.load(models_directories[model_index])
            day_num = 0
            options = {
                'chosen_day': charging_days_list[day_num],
                'charging_network': site,
                'garages': jpl_garages,
                'document': document
            }
            obs, info = eval_env.reset(options=options)
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
                    path_to_directory = fr'testing_days_{site}/charging_day_{ch_day}_{ch_mon}_{ch_year}/'
                    string_representation_of_beta = f"{chosen_beta:.0e}".replace('+', '')
                    new_folder = f'beta={string_representation_of_beta}'
                    folder_path = path_to_directory  + new_folder
                    os.makedirs(folder_path, exist_ok=True)

                    env_path = folder_path + '/'

                    evs_timestamp_reset_list = evs_timestamp_reset[start_testing_for_cycle]
                    end_testing_for_cycle = options['chosen_day'] + timedelta(hours=23, minutes=59, seconds=59)
                    num_of_evs_given_day = len(evs_timestamp_reset_list)
                    ut_signals = np.array(eval_env.chosen_ut_for_each_timestep)
                    cumulative_charging_rates = eval_env.chosen_sum_of_charging_rates
                    cumulative_charging_rates = np.array(cumulative_charging_rates)
                    opt_sum_of_charging_rates = np.sum(offline_optimal_charging_rates[-1, day_num], axis=0)

                    cumulative_costs_per_env[model_index, day_num] = calculate_cumulative_costs_given_ut(
                        uts=ut_signals * (time_between_timesteps / 60),
                        cost_vector=costs_loaded_manually,
                        period=time_between_timesteps)
                    mpes_per_env[model_index, day_num] = mpe_error_fun_rl_testing(
                        ev_diction=eval_env.delivered_and_undelivered_energy)
                    # mses.append(eval_env_current_mse_error)
                    mses_per_env[model_index, day_num] = mse_error_fun_rl_testing(
                        sum_of_charging_rates=cumulative_charging_rates,
                        ut_signals=ut_signals * (time_between_timesteps / 60),
                        capacity_constraint=power_limit*(time_between_timesteps / 60))
                    comparison_of_algs_file = f'comparison_of_ev_charging_algs_beta={string_representation_of_beta}.png'
                    comparison_of_different_algorithms(
                        cumulative_charging_rates=cumulative_charging_rates / (time_between_timesteps / 60),
                        period=time_between_timesteps,
                        opt_signals=opt_sum_of_charging_rates / (time_between_timesteps / 60),
                        path_to_save=env_path + comparison_of_algs_file)


                    pilot_and_real_charging_signal_file = f'comparison_real_pilot_beta={string_representation_of_beta}.png'
                    comparison_pilot_signal_real_signal_graph(ut_signals=ut_signals,
                                                              cumulative_charging_rates=cumulative_charging_rates/(time_between_timesteps/60),
                                                              period=time_between_timesteps,
                                                              path_to_save=env_path + pilot_and_real_charging_signal_file)


                    und_energy_file = f'undelivered_energy_beta={string_representation_of_beta}.txt'
                    undelivered_energy_file_rl(filename=env_path + und_energy_file,
                                               evs_to_undelivered_dict=eval_env.delivered_and_undelivered_energy)

                    ppc_info_filename = f'ppc_inputs_outputs_beta={string_representation_of_beta}.txt'
                    # U = np.linspace(0, eval_env.power_limit, eval_env.power_levels)
                    write_into_file_operator_optimisation(filename=env_path + ppc_info_filename,
                                                          beta=chosen_beta,
                                                          pts=eval_env.normalised_pts,
                                                          U=eval_env.set_that_ut_is_chosen_from,
                                                          generated_uts=eval_env.chosen_ut_for_each_timestep,
                                                          costs_per_u=eval_env.costs_per_u,
                                                          results=eval_env.optim_results_per_u,
                                                          maximum_ramp_rate=eval_env.max_ramp_rate)

                    xts_file = f'xts_values_beta={string_representation_of_beta}.txt'
                    write_xt_states_into_file(filename=env_path + xts_file,
                                                              xts=eval_env.aggregator_states_for_each_timestep)
                    st_ut_file = f'uts_vs_sts_beta={string_representation_of_beta}.txt'
                    write_st_vs_ut_into_file(filename=env_path + st_ut_file,
                                             uts=ut_signals,
                                             sts=cumulative_charging_rates / (time_between_timesteps / 60)
                                             )
                    entrophy_file = f'entrophy_values_beta={string_representation_of_beta}.txt'
                    write_entrophy_to_file(filename=env_path + entrophy_file,
                                           normalised_pts=eval_env.normalised_pts,
                                           entropies=eval_env.entropies_for_each_step)
                    energy_demand_penalty_file = f'not_fully_charged_until_departure_penalty_beta={string_representation_of_beta}.txt'
                    write_energy_demands_penalty(filename=env_path + energy_demand_penalty_file,
                                                 energy_demands_penalties=eval_env.not_fully_charged_before_departure_penalties)
                    table_file_name = f'charging_table_beta={string_representation_of_beta}.csv'
                    create_table(charging_profiles_matrix=eval_env.charging_rates_matrix,
                                 charging_cost_vector=costs_loaded_manually,
                                 period=time_between_timesteps,
                                 show_charging_costs=True,
                                 path_to_save=env_path + table_file_name,
                                 capacity_in_time=ut_signals*(time_between_timesteps/60))


                    day_num += 1
                    if day_num == len(charging_days_list):
                        break
                    options['chosen_day'] = charging_days_list[day_num]
                    evaluation_rewards.append(episode_reward)
                    episode_reward = 0
                    obs, info = eval_env.reset(options=options)
        # reward per each beta in future (possibly) now it works only for 1 beta
        rewards_file = 'evaluation_rewards.txt'
        write_evaluation_rewards_into_file(filename=path_to_outer_directory + rewards_file,
                                           charging_days_list=charging_days_list,
                                           rewards=evaluation_rewards)
        # possibly better to setup these in fixed way and change them everytime like the models too
        # legends_for_each_alg = ['beta = 1e6', 'Offline optimal']
        legends_for_each_env = ['beta = 1e6']
        colors_for_each_env = ['blue']
        path_to_outer_directory = fr'testing_days_{site}/'
        costs_per_alg_settings_file = 'costs_depending_on_parameters.png'
        mpe_file_per_alg_and_day = 'mpe_per_day_for_different_betas.png'

        mpe_per_day_graph(mpe_values_per_alg=mpes_per_env,
                          legend_names_in_order=legends_for_each_env,
                          colors_of_graphs=colors_for_each_env,
                          path_to_save=path_to_outer_directory + mpe_file_per_alg_and_day)
        mse_file_per_alg_and_day = 'mse_per_day_for_different_betas.png'
        mse_per_day_graph(mse_values_per_alg=mses_per_env,
                          legend_names_in_order=legends_for_each_env,
                          colors_of_graphs=colors_for_each_env,
                          path_to_save=path_to_outer_directory + mse_file_per_alg_and_day)
        #
        #
        cumulative_costs_per_algorithm = np.zeros(shape=(number_of_models_tested + 1, k_days))
        cumulative_costs_per_algorithm[0] = deepcopy(cumulative_costs_offline_optimal[-1])
        cumulative_costs_per_algorithm[1] = deepcopy(cumulative_costs_per_env[0])
        # cumulative_costs_per_algorithm[2] = deepcopy(cumulative_costs_per_env[1])
        costs_per_day_graph(costs_per_alg=cumulative_costs_per_algorithm,
                            legend_names_in_order=['offline optimal']+legends_for_each_env,
                            colors_of_graphs=['black']+colors_for_each_env,
                            path_to_save=path_to_outer_directory + costs_per_alg_settings_file)
        #
        # mpe_vs_costs_all_days_file = 'mpe_vs_costs_all_days_per_alg.png'
        # offline_mpes = [1 - 0.1 * i for i in range(len(possible_gammas))]
        # mpe_cost_graph(mpe_values_offline=offline_mpes,
        #                mpe_values_env=None,
        #                cost_values_offline=np.sum(cumulative_costs_offline_optimal, axis=1),
        #                cost_values_env=None,
        #                colors_of_graphs=color_for_each_alg,
        #                legends_of_graphs=legends_for_each_alg,
        #                number_of_days=len(charging_days_list),
        #                path_to_save=path_to_outer_directory + mpe_vs_costs_all_days_file)


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

    # def test_save_all_data_to_files(self):
    #     la_tz = pytz.timezone('America/Los_Angeles')
    #     start_testing = la_tz.localize(datetime(2018, 11, 1, 0, 0, 0))
    #     end_testing = la_tz.localize(datetime(2019, 12, 1, 23, 59, 59))
    #     file = 'jpl_data_training_save.json'
    #     save_data_to_json_via_acn_api(start=start_testing,
    #                                   end=end_testing,
    #                                   site='jpl',
    #                                   path_to_file_save=file)
    #
    #
    #     file = 'caltech_data_training_save.json'
    #     save_data_to_json_via_acn_api(start=start_testing,
    #                                   end=end_testing,
    #                                   site='caltech',
    #                                   path_to_file_save=file)
    #
    #     la_tz = pytz.timezone('America/Los_Angeles')
    #     start_testing = la_tz.localize(datetime(2019, 12, 2, 0, 0, 0))
    #     end_testing = la_tz.localize(datetime(2020, 1, 1, 23, 59, 59))
    #     file = 'caltech_data_testing_save.json'
    #     save_data_to_json_via_acn_api(start=start_testing,
    #                                   end=end_testing,
    #                                   site='caltech',
    #                                   path_to_file_save=file)

    def test_load_jpl_training_data(self):
        # casy su len v timestampoch potrebujem take co su v dnoch preto upravit tu funkciu
        la_tz = pytz.timezone('America/Los_Angeles')
        start_testing = la_tz.localize(datetime(2019, 12, 2, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2020, 1, 1, 23, 59, 59))
        file = 'jpl_data_testing_save.json'
        period = 12
        number_of_evs_interval = [30, np.inf]
        maximum_charging_rate = 7
        evs_timestamp_reset_jpl, evs_timestamp_not_reset_jpl, evs_time_not_normalised_jpl = get_evs_data_from_document_advanced_settings(
            document=file,
            start=start_testing,
            end=end_testing,
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
            period=period,
            allow_overday_charging=False,
            max_charging_rate_within_interval=[maximum_charging_rate,
                                               maximum_charging_rate],
            dates_in_ios_format=True
        )
        # save_data_to_json_via_acn_api(start=start_testing,
        #                               end=end_testing,
        #                               site='jpl',
        #                               path_to_file_save=file)
        plot_hourly_arrivals(evs_time_not_normalised=evs_time_not_normalised_jpl,
                             title='Arrival times for acn static testing data')
        plot_hourly_departures(evs_time_not_normalised=evs_time_not_normalised_jpl,
                               title='Departure times for acn static testing data')
        plot_hourly_requested_energy(evs_time_not_normalised=evs_time_not_normalised_jpl,
                                     title='Energy requested for acn static testing data')

        start_testing = la_tz.localize(datetime(2018, 11, 1, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2019, 12, 1, 23, 59, 59))
        file = 'jpl_data_training_save.json'
        period = 12
        number_of_evs_interval = [30, np.inf]
        maximum_charging_rate = 7
        evs_timestamp_reset_jpl, evs_timestamp_not_reset_jpl, evs_time_not_normalised_jpl = get_evs_data_from_document_advanced_settings(
            document=file,
            start=start_testing,
            end=end_testing,
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
            period=period,
            allow_overday_charging=False,
            max_charging_rate_within_interval=[maximum_charging_rate,
                                               maximum_charging_rate],
            dates_in_ios_format=True
        )
        # save_data_to_json_via_acn_api(start=start_testing,
        #                               end=end_testing,
        #                               site='jpl',
        #                               path_to_file_save=file)
        plot_hourly_arrivals(evs_time_not_normalised=evs_time_not_normalised_jpl,
                             title='Arrival times for acn static testing data')
        plot_hourly_departures(evs_time_not_normalised=evs_time_not_normalised_jpl,
                               title='Departure times for acn static testing data')
        plot_hourly_requested_energy(evs_time_not_normalised=evs_time_not_normalised_jpl,
                                     title='Energy requested for acn static testing data')


        file = 'caltech_data_training_save.json'
        period = 12
        number_of_evs_interval = [30, np.inf]
        maximum_charging_rate = 7
        evs_timestamp_reset_jpl, evs_timestamp_not_reset_jpl, evs_time_not_normalised_jpl = get_evs_data_from_document_advanced_settings(
            document=file,
            start=start_testing,
            end=end_testing,
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
            period=period,
            allow_overday_charging=False,
            max_charging_rate_within_interval=[maximum_charging_rate,
                                               maximum_charging_rate],
            dates_in_ios_format=True
        )
        # save_data_to_json_via_acn_api(start=start_testing,
        #                               end=end_testing,
        #                               site='jpl',
        #                               path_to_file_save=file)
        plot_hourly_arrivals(evs_time_not_normalised=evs_time_not_normalised_jpl,
                             title='Arrival times for acn static testing data')
        plot_hourly_departures(evs_time_not_normalised=evs_time_not_normalised_jpl,
                               title='Departure times for acn static testing data')
        plot_hourly_requested_energy(evs_time_not_normalised=evs_time_not_normalised_jpl,
                                     title='Energy requested for acn static testing data')

        start_testing = la_tz.localize(datetime(2019, 12, 1, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2020, 1, 1, 23, 59, 59))
        file = 'jpl_data_testing_save.json'
        period = 12
        number_of_evs_interval = [30, np.inf]
        maximum_charging_rate = 7
        evs_timestamp_reset_jpl, evs_timestamp_not_reset_jpl, evs_time_not_normalised_jpl = get_evs_data_from_document_advanced_settings(
            document=file,
            start=start_testing,
            end=end_testing,
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
            period=period,
            allow_overday_charging=False,
            max_charging_rate_within_interval=[maximum_charging_rate,
                                               maximum_charging_rate],
            dates_in_ios_format=True
        )
        # save_data_to_json_via_acn_api(start=start_testing,
        #                               end=end_testing,
        #                               site='jpl',
        #                               path_to_file_save=file)
        plot_hourly_arrivals(evs_time_not_normalised=evs_time_not_normalised_jpl,
                             title='Arrival times for acn static testing data')
        plot_hourly_departures(evs_time_not_normalised=evs_time_not_normalised_jpl,
                               title='Departure times for acn static testing data')
        plot_hourly_requested_energy(evs_time_not_normalised=evs_time_not_normalised_jpl,
                                     title='Energy requested for acn static testing data')


    def test_correctness_of_acn_static_data(self):
        la_tz = pytz.timezone('America/Los_Angeles')
        # testing phase
        # start_testing = la_tz.localize(datetime(2019, 12, 2, 0, 0, 0))
        # end_testing = la_tz.localize(datetime(2020, 1, 1, 23, 59, 59))
        maximum_charging_rate = 6.6
        period = 12
        number_of_evse = 54
        number_of_evs_interval = [30, np.inf]
        # # this data must be loaded even if environment loads data separately, to filter charging days
        # evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
        #     charging_network=charging_networks[0],
        #     # garages=caltech_garages,
        #     garages=caltech_garages,
        #     start=start_testing,
        #     end=end_testing,
        #     period=period,
        #     max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
        #     number_of_evs_interval=number_of_evs_interval,
        #     include_weekends=False,
        #     include_overday_charging=False
        # )
        # plot_hourly_arrivals(evs_time_not_normalised=evs_time_not_normalised_time_reset,
        #                      title='Arrival times for acn static testing data')
        # plot_hourly_departures(evs_time_not_normalised=evs_time_not_normalised_time_reset,
        #                      title='Departure times for acn static testing data')
        # plot_hourly_requested_energy(evs_time_not_normalised=evs_time_not_normalised_time_reset,
        #                              title='Energy requested for acn static testing data')
        # document = 'acndata_sessions_testing.json'
        # evs_timestamp_reset2, evs_timestamp_not_reset2, evs_time_not_normalised2 = get_evs_data_from_document_advanced_settings(
        #     document=document,
        #     start=start_testing,
        #     end=end_testing,
        #     number_of_evs_interval=number_of_evs_interval,
        #     include_weekends=False,
        #     period=period,
        #     allow_overday_charging=False,
        #     max_charging_rate_within_interval=[maximum_charging_rate,
        #                                        maximum_charging_rate],
        # )
        # plot_hourly_arrivals(evs_time_not_normalised=evs_time_not_normalised2,
        #                      title='Arrival times for acn testing data')
        # plot_hourly_departures(evs_time_not_normalised=evs_time_not_normalised2,
        #                        title='Departure times for acn  testing data')
        # plot_hourly_requested_energy(evs_time_not_normalised=evs_time_not_normalised2,
        #                              title='Energy requested for acn testing data')
        #
        # start_testing = la_tz.localize(datetime(2018, 11, 1, 0, 0, 0))
        # end_testing = la_tz.localize(datetime(2019, 12, 1, 23, 59, 59))
        #
        # evs_timestamp_reset3, evs_timestamp_not_reset3, evs_time_not_normalised_time_reset3 = load_time_series_ev_data(
        #     charging_network=charging_networks[0],
        #     # garages=caltech_garages,
        #     garages=caltech_garages,
        #     start=start_testing,
        #     end=end_testing,
        #     period=period,
        #     max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
        #     number_of_evs_interval=number_of_evs_interval,
        #     include_weekends=False,
        #     include_overday_charging=False
        # )
        # plot_hourly_arrivals(evs_time_not_normalised=evs_time_not_normalised_time_reset3,
        #                      title='Arrival times for acn static training data')
        # plot_hourly_departures(evs_time_not_normalised=evs_time_not_normalised_time_reset3,
        #                        title='Departure times for acn static training data')
        #
        # plot_hourly_requested_energy(evs_time_not_normalised=evs_time_not_normalised_time_reset3,
        #                              title='Energy requested for acn static training data')
        # document1 = 'acndata_sessions_jpl.json'
        # start_testing = la_tz.localize(datetime(2019, 12, 2, 0, 0, 0))
        # end_testing = la_tz.localize(datetime(2020, 1, 1, 23, 59, 59))
        # evs_timestamp_reset_jpl, evs_timestamp_not_reset_jpl, evs_time_not_normalised_jpl = get_evs_data_from_document_advanced_settings(
        #     document=document1,
        #     start=start_testing,
        #     end=end_testing,
        #     number_of_evs_interval=number_of_evs_interval,
        #     include_weekends=False,
        #     period=period,
        #     allow_overday_charging=False,
        #     max_charging_rate_within_interval=[maximum_charging_rate,
        #                                        maximum_charging_rate],
        # )
        # plot_hourly_arrivals(evs_time_not_normalised=evs_time_not_normalised_jpl,
        #                      title='Arrival times for acn testing data')
        # plot_hourly_departures(evs_time_not_normalised=evs_time_not_normalised_jpl,
        #                        title='Departure times for acn  testing data')
        # plot_hourly_requested_energy(evs_time_not_normalised=evs_time_not_normalised_jpl,
        #                              title='Energy requested for acn testing data')

        document2 = 'acndata_sessions_jpl_training.json'
        start_testing = la_tz.localize(datetime(2018, 11, 1, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2019, 12, 1, 23, 59, 59))

        evs_timestamp_reset_jpl, evs_timestamp_not_reset_jpl, evs_time_not_normalised_jpl = get_evs_data_from_document_advanced_settings(
            document=document2,
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
    # def test_is_feasible_testing(self):
    #     la_tz = pytz.timezone('America/Los_Angeles')
    #     # testing phase
    #     start_testing = la_tz.localize(datetime(2019, 12, 2, 0, 0, 0))
    #     end_testing = la_tz.localize(datetime(2020, 1, 1, 23, 59, 59))
    #     maximum_charging_rate = 3
    #     period = 12
    #     number_of_evse = 54
    #     number_of_evs_interval = [30, np.inf]
    #     # this data must be loaded even if environment loads data separately, to filter charging days
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
    #         include_overday_charging=False
    #     )
    #     cost_file = '20241025-20241025 CAISO Day-Ahead Price (1).csv'
    #     # costs_loaded_manually = load_locational_marginal_prices(filename=cost_file, organization='VEA',period=period)
    #     costs_loaded_manually = load_locational_marginal_prices(filename=cost_file, organization=None, period=period)
    #     for charging_date, evs in  evs_timestamp_reset.items():
    #         end_charging_date = charging_date + timedelta(hours=23,minutes=59, seconds=59)
    #         scheduling_alg = OPT(EVs=evs,
    #                              start=charging_date,
    #                              end=end_charging_date,
    #                              available_energy_for_each_timestep=0,
    #                              ut_interval=[0, 150],
    #                              time_between_timesteps=period,
    #                              number_of_evse=number_of_evse,
    #                              costs_loaded_manually=costs_loaded_manually)
    #         feasibility, opt_charging_rates = scheduling_alg.solve()
    #         self.assertTrue(feasibility)
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


    # TODO: currently doing this test
    def testRLourenvlibrary(self):
        la_tz = pytz.timezone('America/Los_Angeles')
        start_testing = la_tz.localize(datetime(2018, 11, 1, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2019, 12, 1, 23, 59, 59))
        # maximum_charging_rate = 6.6
        # maximum_charging_rate = 7
        maximum_charging_rate = 10
        period = 12
        max_number_of_episodes = 500
        number_of_timesteps_in_one_episode = ((60 * 24) / period)
        total_timesteps = max_number_of_episodes * number_of_timesteps_in_one_episode
        number_of_evs_interval = [30, np.inf]
        document_caltech = 'caltech_data_training_save.json'

        evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = get_evs_data_from_document_advanced_settings(
            document=document_caltech,
            start=start_testing,
            end=end_testing,
            period=period,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
            allow_overday_charging=False,
            dates_in_ios_format=True
        )

        charging_days_list_caltech = list(evs_timestamp_reset.keys())
        document_jpl = 'jpl_data_training_save.json'

        evs_timestamp_reset_jpl, evs_timestamp_not_reset_jpl, evs_time_not_normalised_time_reset_jpl = get_evs_data_from_document_advanced_settings(
            document=document_jpl,
            start=start_testing,
            end=end_testing,
            period=period,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
            allow_overday_charging=False,
            dates_in_ios_format=True
        )
        charging_days_list_jpl = list(evs_timestamp_reset_jpl.keys())
        charging_days_per_station_list = [charging_days_list_jpl, charging_days_list_caltech]
        # charging_days_per_station_list = [charging_days_list_caltech, charging_days_list_jpl]
        # total_timesteps = 4000000
        total_timesteps = 2000000
        # total_timesteps = 240000
        # scheduling_algorithm = LeastLaxityFirstAlg
        scheduling_algorithm = SmoothedLeastLaxityAlg
        # beta = 1e6
        beta=1e6
        ts = np.arange(0, 24, (period/60))
        costs_loaded_manually = [(1-(ts[i]/24)) for i in range(len(ts))]
        power_limit = 150
        # power_levels = 20
        power_levels = 10
        ramp_rate = 30
        env = EVenvironment(scheduling_algorithm=scheduling_algorithm,
                            time_between_timesteps=period,
                            tuning_parameter=beta,
                            max_charging_rate=maximum_charging_rate,
                            cost_list=costs_loaded_manually,
                            power_levels=power_levels,
                            power_limit=power_limit,
                            charging_stations=['jpl','caltech'],
                            charging_days_per_charging_station=charging_days_per_station_list,
                            data_files=[document_jpl,document_caltech],
                            max_ramp_rate=ramp_rate,
                            costs_in_kwh=False)

        model = SAC("MlpPolicy",
                    env,
                    # gamma=0.5,
                    gamma=0.5,
                    ent_coef=0.5,
                    tensorboard_log="./sac/",
                    verbose=0)
        # reward_function_form = ''
        model.learn(total_timesteps=total_timesteps, callback=TensorboardCallback())

        model_name = 'sac_50'
        dir_where_saved_models_are = 'SavedModels/ev_experiments1/'
        model.save(dir_where_saved_models_are+model_name)

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
        filename = 'caiso_2016/acndata_sessions_acn.json'
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