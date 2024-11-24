import copy
import math
import random
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from stable_baselines3.common.logger import Figure
from cvxpy import SolverError
import gzip, csv
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
import os
import pytz
from acnportal.acndata import DataClient
from networks import *
import pkg_resources
import importlib.resources
from pathlib import Path
# source: https://stackoverflow.com/questions/3703276/how-to-tell-if-a-file-is-gzip-compressed
def is_gzipped(path):
    with open(path, "rb") as f:
        return f.read(2) == b'\x1f\x8b'

def check_package_exists(package_name):
    try:
        # Attempt to get the distribution of the package
        dist = pkg_resources.get_distribution(package_name)
        print(f'The package "{package_name}" is installed.')
        print(f'Location: {dist.location}')
    except pkg_resources.DistributionNotFound:
        print(f'The package "{package_name}" is not installed.')

# the connection time should be between start and end
def check_date_within_interval(file_date:datetime, start:datetime, end:datetime):
    if start <= file_date <= end:
        return True
    return False

def is_weekend(date):
    # Returns True if the date is a Saturday or Sunday, False otherwise
    return date.weekday() >= 5

def check_time_series_file_correctness(file_name,
                                       start:datetime,
                                       end:datetime,
                                       include_weekends=False,
                                       ):
    # all times are converted to Los Angeles local timezone
    # only files with this format are accepted, and the further checking of gzipped files is elsewhere
    if not file_name.endswith('.csv.gz'):
        return False
    splitted_string = file_name.split('-')[4:-1]
    file_date_arr = splitted_string[:2] + splitted_string[2].split('T') + splitted_string[3:]

    la_lt = pytz.timezone('America/Los_Angeles')
    year, month, day, hour, minute, second = file_date_arr
    # utc is default timezone
    # name of file contains the info about arrivals in utc timezone
    file_date = datetime(int(year),
                         int(month),
                         int(day),
                         int(hour),
                         int(minute),
                         int(second),
                         tzinfo=timezone.utc)
    file_date = file_date.astimezone(la_lt)

    if not check_date_within_interval(file_date=file_date,
                                      start=start,
                                      end=end):
        return False

    if is_weekend(date=file_date) and not include_weekends:
        return False

    return True


def datetime_to_timestamp(start, chosen_date, period, round_up=False):
    ts = (chosen_date - start) / timedelta(minutes=period)
    if round_up:
        return int(math.ceil(ts))
    else:
        return int(ts)

def read_and_extract_time_series_ev_info(file,
                                         start,
                                         period,
                                         max_charging_rate_within_interval=None,
                                         include_overday_charging=True):

    lines = None
    try:
        lines = file.readlines()
    except gzip.BadGzipFile:
        return False, None, None

    # first line is notation
    # second line should be start of charging
    # last line is end of charging possibly
    # the charging sessions with one timestep are useless and probably not included in authors implementation
    if len(lines) < 3:
         return False, None, None

    # Get the last and second last lines
    second_line_start_charging_date = datetime.fromisoformat(lines[1].split(',')[0])
    last_line_end_charging_date = datetime.fromisoformat(lines[-1].split(',')[0])
    second_last_line_delivered_energy = 0
    # if lines[-2].split(',')[energy_index] != '':
    #     second_last_line_delivered_energy = float(lines[-2].split(',')[energy_index])

    #
    for i in range(len(lines)-1, 0, -1):
        state = lines[i].split(',')[4]
        if i == len(lines)-1 and state != 'UNPLUGGED':
            return False, None, None
        if state != 'UNPLUGGED':
            energy_delivered = lines[i].split(',')[5]
            if len(energy_delivered) == 0:
                second_last_line_delivered_energy = 0
                break
            second_last_line_delivered_energy = float(energy_delivered)
            break

    # delete evs that have not charged anything
    if second_last_line_delivered_energy == 0:
        return False, None, None
    la_lt = pytz.timezone('America/Los_Angeles')
    start_of_day = second_line_start_charging_date.astimezone(la_lt).replace(hour=0, minute=0, second=0, microsecond=0)
    arrival_time = second_line_start_charging_date.astimezone(la_lt)
    departure_time = last_line_end_charging_date.astimezone(la_lt)
    energy_requested = second_last_line_delivered_energy

    arrival_timestamp_reseted = datetime_to_timestamp(start=start_of_day, chosen_date=arrival_time, period=period)
    departure_timestamp_reseted = datetime_to_timestamp(start=start_of_day, chosen_date=departure_time, period=period)

    max_timestamp = int((24*60)/period) - 1

    if include_overday_charging == False and arrival_time.date() != departure_time.date():
        return False, None, None
    if departure_timestamp_reseted > max_timestamp:
        ...
    arrival_timestamp = datetime_to_timestamp(start=start, chosen_date=arrival_time,
                                              period=period)
    departure_timestamp = datetime_to_timestamp(start=start, chosen_date=departure_time,
                                                period=period)


    max_charging_rate = 6.6

    if max_charging_rate_within_interval is not None:
        max_charging_rate = random.uniform(max_charging_rate_within_interval[0],
                                           max_charging_rate_within_interval[1])
    # we assume we are able to deliver enough energy to evs
    # less conservative
    if (1 + (departure_timestamp_reseted - arrival_timestamp_reseted))*max_charging_rate < energy_requested:
        ...
        # raise ValueError('Maximum charging rate is too low to satisfy EVs demands')
    res_ev_reseted = [start_of_day,
                  arrival_time,
                  arrival_timestamp_reseted,
                  departure_time,
                  departure_timestamp_reseted,
                  max_charging_rate,
                  energy_requested]
    res_ev_not_reseted = [start_of_day,
                  arrival_time,
                  arrival_timestamp,
                  departure_time,
                  departure_timestamp,
                  max_charging_rate,
                  energy_requested]
    return True, res_ev_reseted, res_ev_not_reseted
# used for testing and training purposes of RL algorithm
# preferrably with bigger amount of cars, so it is undesirable to choose random n cars, RL algorithm might not work then
# TODO: fix rest of algorithm, but for now it is better to leave it like that for test purposes
# if we want random n of evs we can choose it outside of the function
def filter_evs(date_to_evs_diction_timestamp_reseted:dict,
               date_to_evs_diction_timestamp_not_reseted:dict,
               date_to_evs_diction_time_not_normalised:dict,
               number_of_evs_interval):

    res_diction_timestamp_reseted = {}
    res_diction_timestamp_not_reseted = {}
    res_diction_time_not_normalised = {}

    index_of_ev = 0
    # index_of_ev_reset_after_day = 0
    if number_of_evs_interval is not None:
        for key, value in date_to_evs_diction_timestamp_reseted.items():
            if number_of_evs_interval[0] <= len(value) < number_of_evs_interval[1]:
                ...
            else:
                continue
            if res_diction_timestamp_reseted.get(key) is None:
                res_diction_timestamp_reseted[key] = []
                res_diction_timestamp_not_reseted[key] = []
                res_diction_time_not_normalised[key] = []

            for index_of_ev_reset_after_day, ev in enumerate(value, start=0):
                res_diction_timestamp_reseted[key].append([index_of_ev_reset_after_day] + ev)
                res_diction_timestamp_not_reseted[key].append([index_of_ev] + date_to_evs_diction_timestamp_not_reseted[key][index_of_ev_reset_after_day])
                res_diction_time_not_normalised[key].append([index_of_ev] + date_to_evs_diction_time_not_normalised[key][index_of_ev_reset_after_day])
                index_of_ev += 1
    else:
        res_diction_timestamp_reseted = deepcopy(date_to_evs_diction_timestamp_reseted)
        res_diction_timestamp_not_reseted = deepcopy(date_to_evs_diction_timestamp_not_reseted)
        res_diction_time_not_normalised = deepcopy(date_to_evs_diction_time_not_normalised)

    return res_diction_timestamp_reseted, res_diction_timestamp_not_reseted, res_diction_time_not_normalised
def convert_evs_diction_to_array(evs_diction):
    res = []
    for key, value in evs_diction.items():
        for ev in value:
            res.append(ev)
    return res

def load_time_series_ev_data(charging_network:str,
                             garages:list,
                             start:datetime,
                             end:datetime,
                             period,
                             number_of_evs_interval=None,
                             max_charging_rate_within_interval=None,
                             include_weekends=False,
                             include_overday_charging=True,
                             dir_where_is_projected_cloned =fr'C:\Users\OMI\Documents\thesisdata'):


    charging_netw_dict = {charging_networks[0]: caltech_garages,
                          charging_networks[1]: jpl_garages,
                          charging_networks[2]: office_01_garages}

    if charging_netw_dict.get(charging_network) == None:
        raise KeyError(f'Charging network with name {charging_network} does not exist!\n')
    if not set(garages).issubset(set(charging_netw_dict[charging_network])):
        raise ValueError(f'Charging network with name {charging_network} does not have at least one garage from list: {garages}!\n')


    date_to_evs_diction_timestamp_reseted = {}
    date_to_evs_diction_timestamp_not_reseted = {}
    date_to_evs_diction_time_not_normalised = {}

    # check_package_exists('ACNDataStatic')
    for i in range(len(garages)):
        path_to_files = dir_where_is_projected_cloned + fr'\ACN-Data-Static\time series data\{charging_network}\{garages[i]}'
        package_dir = Path(path_to_files)
        # Check if the subdirectory exists
        if os.path.isdir(package_dir):
            # List all files in the subdirectory
            for csv_file in package_dir.iterdir():
                file_name = csv_file.name

                if not check_time_series_file_correctness(
                        file_name=file_name,
                        start=start,
                        end=end,
                        include_weekends=include_weekends):
                    continue

                if not is_gzipped(csv_file):
                    continue

                with gzip.open(csv_file, 'rt') as csv_file_opened:
                    # Read all lines from the file
                    value  = read_and_extract_time_series_ev_info(
                        file=csv_file_opened,
                        start=start,
                        period=period,
                        max_charging_rate_within_interval=max_charging_rate_within_interval,
                        include_overday_charging=include_overday_charging)

                    valid, extracted_ev_info_reseted, extracted_ev_info_not_reseted  = value[0], value[1], value[2]

                    if valid is False:
                        continue

                    start_of_day, arrival_time, arrival_timestamp_reseted, departure_time, departure_timestamp_reseted, max_charging_rate, energy_requested = extracted_ev_info_reseted
                    _, _, arrival_timestamp_not_reseted, _, departure_timestamp_not_reseted, _, _ = extracted_ev_info_not_reseted

                    ev_reseted = [arrival_timestamp_reseted,
                          departure_timestamp_reseted,
                          max_charging_rate,
                          energy_requested]
                    ev_not_reseted = [arrival_timestamp_not_reseted,
                          departure_timestamp_not_reseted,
                          max_charging_rate,
                          energy_requested]

                    ev_time_not_normalised = [arrival_time,
                                              departure_time,
                                              max_charging_rate,
                                              energy_requested]

                    if date_to_evs_diction_timestamp_reseted.get(start_of_day) is None:
                        date_to_evs_diction_timestamp_reseted[start_of_day] = []
                        date_to_evs_diction_time_not_normalised[start_of_day] = []
                        date_to_evs_diction_timestamp_not_reseted[start_of_day] = []

                    date_to_evs_diction_timestamp_reseted[start_of_day].append(ev_reseted)
                    date_to_evs_diction_time_not_normalised[start_of_day].append(ev_time_not_normalised)
                    date_to_evs_diction_timestamp_not_reseted[start_of_day].append(ev_not_reseted)

        else:
            # print(f"Subdirectory '{subdirectory}' does not exist.")
            return ValueError(f"Subdirectory '{path_to_files}' does not exist.")
        res_diction_timestamp_reseted, res_diction_timestamp_not_reseted, res_diction_time_not_normalised = filter_evs(
            date_to_evs_diction_timestamp_reseted=date_to_evs_diction_timestamp_reseted,
            date_to_evs_diction_timestamp_not_reseted=date_to_evs_diction_timestamp_not_reseted,
            date_to_evs_diction_time_not_normalised=date_to_evs_diction_time_not_normalised,
            number_of_evs_interval=number_of_evs_interval)

        return res_diction_timestamp_reseted, res_diction_timestamp_not_reseted, res_diction_time_not_normalised

# seems the costs are cumulative but i dont understand why they have such value
def cost_values_per_day(cost_values):
    x_values = range(len(cost_values))

    # Plotting the MSE values
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, cost_values, marker='o', linestyle='-', color='b')

    # Adding labels and title
    # plt.title('Mean Squared Error per Day')
    plt.xlabel('Dni')
    # find slovak translation of MSE
    plt.ylabel('Cena')
    # plt.xticks(x_values)  # Set x ticks to be the indices
    plt.grid()
    plt.legend()

    plt.ylim(bottom=0)
    plt.xlim(left=0, right=1)

    # Show the plot
    plt.show()
# can work for
def mpe_cost_graph(
        mpe_values_offline,
        mpe_values_env,
        cost_values_offline,
        cost_values_env,
        colors_of_graphs,
        legends_of_graphs,
        number_of_days=14,
        path_to_save=''):
    # mpe_values_per_alg,cost_values_per_alg two dimensional array
    # in case of offline and mpc algorithm gamma = 1 - mpe
    # trade off between mpe and cumulative costs (not averaged)


    plt.figure(figsize=(10, 5))
    for i in range(2):

        mpe_values_of_given_alg = []
        costs_of_given_alg = []
        if i == 0 and mpe_values_env is not None and cost_values_env is not None:
            mpe_values_of_given_alg = mpe_values_env
            costs_of_given_alg = cost_values_env / number_of_days
        elif i == 1 and mpe_values_offline is not None and cost_values_offline is not None:
            mpe_values_of_given_alg = mpe_values_offline
            costs_of_given_alg = cost_values_offline / number_of_days
        elif len(mpe_values_of_given_alg) == 0 and len(costs_of_given_alg) == 0:
            continue
        plt.plot(mpe_values_of_given_alg, costs_of_given_alg,
                 marker='o', linestyle='-', color=colors_of_graphs[i])


    # Adding labels and title
    # plt.title('Mean Squared Error per Day')
    plt.xlabel('MPE')
    plt.ylabel('Priemerné kumulatívne ceny')
    # plt.xticks(x_values)  # Set x ticks to be the indices
    plt.grid()
    plt.legend(legends_of_graphs)

    plt.ylim(bottom=0)
    plt.xlim(left=0, right=1)
    if len(path_to_save) == 0:
        # Show the plot
        plt.show()
    else:
        plt.savefig(path_to_save)
        plt.close()


def get_ut_signals_from_schedule(schedule:np.array):
    cols = schedule.shape[1]
    uts = []
    for i in range(cols):
        uts.append(math.fsum(schedule[:, i]))
    return uts

def plot_arrivals_for_given_day(evs, day, period):
    x_values = np.arange(0, 24, period / 60)
    timesteps_during_the_day = int((24*60)/period)
    y_weights = [0 for i in range(timesteps_during_the_day)]
    for i in range(len(evs)):
        index, arrival_timestamp, departure_timestamp, maximum_charging_rate, energy_requested = evs[i]
        y_weights[arrival_timestamp] += 1


    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_weights, marker='o', linestyle='-', color='b')

    # Adding labels and title
    plt.title(f'Príchody elektromobilov pre deň {day}')
    # plt.title('Mean Squared Error per Day')
    plt.xlabel('Hodiny')
    plt.ylabel('Počet príchodov elektromobilov')
    # plt.xticks(x_values)  # Set x ticks to be the indices
    plt.xticks(np.arange(0, 25, 1))
    plt.grid()
    plt.legend()

    plt.ylim(bottom=0)
    plt.xlim(left=0, right=24)
    # Show the plot
    plt.show()

def plot_departures_for_given_day(evs, day, period):
    x_values = np.arange(0, 24, period / 60)
    timesteps_during_the_day = int((24 * 60) / period)
    y_weights = [0 for i in range(timesteps_during_the_day)]
    for i in range(len(evs)):
        index, arrival_timestamp, departure_timestamp, maximum_charging_rate, energy_requested = evs[i]
        if departure_timestamp >= len(y_weights):
            continue
        y_weights[departure_timestamp] += 1
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_weights, marker='o', linestyle='-', color='b')

    plt.title(f'Odchody elektromobilov pre deň {day}')
    # Adding labels and title
    # plt.title('Mean Squared Error per Day')
    plt.xlabel('Hodiny')
    plt.ylabel('Počet odchodov elektromobilov')
    # plt.xticks(x_values)  # Set x ticks to be the indices
    plt.xticks(np.arange(0, 25, 1))
    plt.grid()
    plt.legend()

    plt.ylim(bottom=0)
    plt.xlim(left=0, right=24)
    # Show the plot
    plt.show()
#
def convert_mwh_to_kwh_prices(prices, time_between_timesteps):
    return (prices / 1000)

def comparison_pilot_signal_real_signal_graph(ut_signals,
                                              cumulative_charging_rates,
                                              period,
                                              path_to_save=''):
    x_values = np.arange(0, 24, (period/60))
    plt.figure(figsize=(10, 5))
    divisor = (period/60)
    plt.plot(x_values, ut_signals, marker='o', linestyle='-', color='b')
    plt.plot(x_values, cumulative_charging_rates, marker='o', linestyle='-', color='r')
    # if opt_signals is not None:
    #     plt.plot(x_values, opt_signals, marker='o', linestyle='-', color='k')
    # Adding labels and title
    # plt.title('Mean Squared Error per Day')
    plt.xlabel('Hodiny')
    plt.ylabel('Sila (kW)')
    # plt.xticks(x_values)  # Set x ticks to be the indices
    plt.xticks(np.arange(0, 25, 1))
    plt.grid()
    plt.legend(['Hodnoty ut signálu', 'Hodnoty st plánu nabíjania'], fontsize='medium')

    plt.ylim(bottom=0)
    plt.xlim(left=0, right=24)
    if len(path_to_save) == 0:
        plt.show()
    else:
        plt.savefig(path_to_save)
        plt.close()
    # Show the plot
    #

def comparison_of_pilot_signal_real_signal_graph_kwh(ut_signals,
                                                     cumulative_charging_rates,
                                                     period,
                                                     path_to_save=''):
    x_values = np.arange(0, 24, (period / 60))
    plt.figure(figsize=(10, 5))
    divisor = (period / 60)
    new_ut_signals = np.array(ut_signals) / divisor
    new_cumulative_charging_rates = np.array(cumulative_charging_rates) / divisor


    ut_signals_kwh = np.zeros(shape=(len(new_ut_signals)))
    cumulative_charging_rates_kwh = np.zeros(shape=(len(cumulative_charging_rates)))
    num_of_signals_per_hour = int(60/(period))
    for i in range(len(ut_signals)):
        if i == 0:
            ut_signals_kwh[i] = 0
            cumulative_charging_rates_kwh[i] = 0
        ut_signals_kwh[i] = np.mean(new_ut_signals[i-num_of_signals_per_hour:i])
        cumulative_charging_rates_kwh[i] = np.mean(new_cumulative_charging_rates[i-num_of_signals_per_hour:i])
    plt.plot(x_values, ut_signals_kwh, marker='o', linestyle='-', color='b')
    plt.plot(x_values, cumulative_charging_rates_kwh, marker='o', linestyle='-', color='r')
    # if opt_signals is not None:
    #     plt.plot(x_values, opt_signals, marker='o', linestyle='-', color='k')
    # Adding labels and title
    # plt.title('Mean Squared Error per Day')
    plt.xlabel('Hodiny')
    plt.ylabel('Energia (kWh)')
    # plt.xticks(x_values)  # Set x ticks to be the indices
    plt.xticks(np.arange(0, 25, 1))
    plt.grid()
    plt.legend(['Hodnoty ut signálu', 'Množstvá dodanej energie pre všetky t'], fontsize='medium')

    plt.ylim(bottom=0)
    plt.xlim(left=0, right=24)
    if len(path_to_save) == 0:
        plt.show()
    else:
        plt.savefig(path_to_save)
        plt.close()
def comparison_of_different_algorithms(cumulative_charging_rates,
                                              period,
                                              opt_signals,
                                              path_to_save=''):
    x_values = np.arange(0, 24, (period / 60))
    divisor = (period/60)
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, np.array(cumulative_charging_rates), marker='o', linestyle='-', color='r')
    if opt_signals is not None:
        plt.plot(x_values, np.array(opt_signals), marker='o', linestyle='-', color='k')
    # Adding labels and title
    # plt.title('Mean Squared Error per Day')
    plt.xlabel('Hodiny')
    plt.ylabel('Sila (kW)')
    # plt.xticks(x_values)  # Set x ticks to be the indices
    plt.xticks(np.arange(0, 25, 1))
    plt.grid()
    plt.legend([ 'PPC','Offline optimal'], fontsize='medium')

    plt.ylim(bottom=0)
    plt.xlim(left=0, right=24)
    if len(path_to_save) == 0:
        plt.show()
    else:
        plt.savefig(path_to_save)
        # Show the plot
        plt.close()
# seems the costs are cumulative but i dont understand why they have such value
def charging_in_time_graph(ut_signals_offline,
                           period,
                           ut_signals_ppc=None,
                           ut_signals_mpc=None):
    # x_values = range(len(ut_signals_offline))
    x_values = np.arange(0, 24, period/60)
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, ut_signals_offline, marker='o', linestyle='-', color='b')

    # Adding labels and title
    # plt.title('Mean Squared Error per Day')
    plt.xlabel('Hodiny')
    plt.ylabel('Sila (kW)')
    # plt.xticks(x_values)  # Set x ticks to be the indices
    plt.xticks(np.arange(0, 25, 1))
    plt.grid()
    plt.legend()

    plt.ylim(bottom=0)
    plt.xlim(left=0, right=24)
    # Show the plot
    plt.show()
def convert_kw_to_mwh(signal, period):
    return (signal * (period/60))/1000
def plot_costs(costs,period,
               ticks_after_hours=1,
               path_to_save=''):
    x_values = np.arange(0, 24, period / 60)
    # Plotting the MSE values
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, costs, linestyle='-', color='b')

    # Adding labels and title
    # plt.title('Mean Squared Error per Day')
    plt.xlabel('Hodiny')
    # find slovak translation of MSE
    plt.ylabel('Cena ($/MWh)')

    plt.xticks(np.arange(0, 25, 1))

    # plt.xticks(x_values)  # Set x ticks to be the indices
    plt.grid()
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlim(left=0, right=24)
    # Show the plot
    if len(path_to_save) == 0:
        plt.show()
    else:
        plt.savefig(path_to_save)
        plt.close()

# caclulates for opt not for rl algorithm
# assumes that schedule is a matrix where column is charging plan for time t
def calculate_cumulative_costs(schedule:np.array,cost_vector):
    res = 0
    overall_sum_of_charging_rates = 0
    cols = schedule.shape[1]
    for i in range(cols):
            signal = math.fsum(schedule[:, i])
            res += signal * cost_vector[i]
        # if lmp:
        #     signal = math.fsum(schedule[:,i])
        #     # res += ((signal*(60/period))/1000) * cost_vector[i]
        #     res += ((signal * (60 / period)) / 1000) * cost_vector[i]
        # else:
        #     res += math.fsum(cost_vector[i] * schedule[:, i])
    return res

def calculate_cumulative_costs_given_ut(uts:np.array,cost_vector, period=12):
    res = 0
    for i, ut in enumerate(uts, start=0):
            res += ut * cost_vector[i]
    return res

# assumes that costs_per_alg is 2 dimensional and each element are costs for specific alg
def costs_per_day_graph(costs_per_alg,
                        legend_names_in_order,
                        colors_of_graphs,
                        path_to_save=''):
    costs_per_alg = np.array(costs_per_alg)
    if costs_per_alg.ndim == 1:
        costs_per_alg = [costs_per_alg]
    plt.figure(figsize=(10, 5))
    x_values = range(len(costs_per_alg[0]))
    for i in range(len(costs_per_alg)):
        plt.plot(x_values, costs_per_alg[i], marker='o', linestyle='-', color=colors_of_graphs[i])

    # Adding labels and title
    # plt.title('Mean Squared Error per Day')
    plt.xlabel('Dni')
    # find slovak translation of MSE
    plt.ylabel('Kumulatívna ceny')
    plt.xticks(x_values)  # Set x ticks to be the indices
    plt.grid()
    plt.legend(legend_names_in_order)

    plt.ylim(bottom=0)
    if len(path_to_save) == 0:
        # Show the plot
        plt.show()
    else:
        plt.savefig(path_to_save)
        plt.close()
def mpe_per_day_graph(mpe_values_per_alg,
                      legend_names_in_order,
                      colors_of_graphs,
                      path_to_save='',
                      percentual=True
                      ):
    mpe_values_per_alg = np.array(mpe_values_per_alg) if not percentual else np.array(mpe_values_per_alg)*100
    if mpe_values_per_alg.ndim == 1:
        mpe_values_per_alg = [mpe_values_per_alg]
    plt.figure(figsize=(10, 5))
    x_values = range(len(mpe_values_per_alg[0]))
    for i in range(len(mpe_values_per_alg)):
        plt.plot(x_values, mpe_values_per_alg[i], marker='o', linestyle='-', color=colors_of_graphs[i])


    # Create an array of indices (x values)
    # x_values = range(len(mpe_values_per_alg))

    # Plotting the MSE values
    # plt.figure(figsize=(10, 5))
    # plt.plot(x_values, mpe_values, marker='o', linestyle='-', color='b')

    # Adding labels and title
    # plt.title('Mean Squared Error per Day')
    plt.xlabel('Dni')
    # find slovak translation of MSE
    if percentual:
        plt.ylabel('MPE (%)')
    if not percentual:
        plt.ylabel('MPE')
    plt.xticks(x_values)  # Set x ticks to be the indices
    plt.grid()
    plt.legend(legend_names_in_order)

    # Set y-axis limits from 0 to 1
    if not percentual:
        plt.ylim(0, 1)
    if percentual:
        plt.ylim(0, 100)
    if len(path_to_save) == 0:
        # Show the plot
        plt.show()
    else:
        plt.savefig(path_to_save)
        plt.close()
def mse_per_day_graph(mse_values_per_alg,
                      legend_names_in_order,
                      colors_of_graphs,
                      path_to_save='',
                      percentual=True):
    # Create an array of indices (x values)

    # Plotting the MSE values
    plt.figure(figsize=(10, 5))
    # plt.plot(x_values, mse_values, marker='o', linestyle='-', color='b', label='MSE Values')

    mse_values_per_alg = np.array(mse_values_per_alg) if not percentual else np.array(mse_values_per_alg) *100
    if mse_values_per_alg.ndim == 1:
        mse_values_per_alg = [mse_values_per_alg]
    plt.figure(figsize=(10, 5))
    x_values = range(len(mse_values_per_alg[0]))
    for i in range(len(mse_values_per_alg)):
        plt.plot(x_values, mse_values_per_alg[i], marker='o', linestyle='-', color=colors_of_graphs[i])

    # Adding labels and title
    # plt.title('Mean Squared Error per Day')
    plt.xlabel('Dni')
    # find slovak translation of MSE
    if not percentual:
        plt.ylabel('MSE')
    if percentual:
        plt.ylabel('MSE (%)')
    plt.xticks(x_values)  # Set x ticks to be the indices
    plt.grid()
    plt.legend(legend_names_in_order)

    # Set y-axis limits from 0 to 1
    # plt.ylim(0, 1)
    # Show the plot
    if len(path_to_save) == 0:
        plt.show()
    else:
        plt.savefig(path_to_save)
        plt.close()
# draws barchart for one day
def draw_barchart_sessions_from_RL(dict_of_evs,
                                   dict_of_arrivals_and_departures,
                                   tick_after_bars=10,
                                   charging_date=None,
                                   alg='PPC',
                                   path_to_save=''):
    energies_requested = []
    energies_undelivered = []
    for key, value in dict_of_evs.items():
        energy_requested, energy_undelivered = value
        energies_requested.append(energy_requested)
        energies_undelivered.append(energy_undelivered)
    plt.bar([i for i in range(len(energies_requested))], energies_requested, color='lightblue', edgecolor='lightblue')
    plt.bar([i for i in range(len(energies_requested))], energies_undelivered, color='black', edgecolor='black')
    # Add titles and labels
    # plt.title('Sample Bar Chart', fontsize=16)
    plt.xlabel('Nabíjanie elektromobily', fontsize=12)
    plt.ylabel('Energia (kWh)', fontsize=12)

    # Add gridlines
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    num_bars = len(energies_requested)
    tick_positions = list(range(0, num_bars, tick_after_bars))  # Positions for labels (19, 39, 59, ...)
    custom_labels = [f'{tick_after_bars * i}' for i in range(len(tick_positions))]

    # Set ticks and labels
    plt.xticks(ticks=tick_positions, labels=custom_labels)

    # Add a legend with comments only
    plt.legend(['dodaná energia', 'nedodaná energia'], fontsize='medium')
    if charging_date is None:
        plt.title(f'Dodaná energia elektromobilom pri použití {alg} algoritmu')
    else:
        formatted_date = charging_date.strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
        plt.title(f'Dodaná energia elektromobilom (deň: {formatted_date}) pri použití {alg} algoritmu')
    if len(path_to_save) == 0:
        # Show the plot
        plt.show()
    else:
        plt.savefig(path_to_save)
        plt.close()


# assume that evs is dict (it can change) possibly not because we plot for specific day so key is important
def draw_barchart_sessions(schedule:np.array, evs_dict_reseted,
                           tick_after_bars=10):
    # Data
    energies_requested_for_each_day = {}
    energies_undelivered_for_each_day = {}
    for key, value in evs_dict_reseted.items():
        energies_requested_for_each_day[key] = []
        energies_undelivered_for_each_day[key] = []
        for ev in value:
            ev_index, arrival_timestamp_reseted, departure_timestamp_reseted, max_charging_rate, energy_requested = ev
            energies_requested_for_each_day[key].append(energy_requested)
            try:
                energy_undelivered = energy_requested - math.fsum(schedule[ev_index,:])
                energies_undelivered_for_each_day[key].append(max(energy_undelivered, 0))
            except:
                energies_undelivered_for_each_day[key].append(max(energy_requested - 0, 0))


    # Create a bar chart
    for key, value in energies_requested_for_each_day.items():
        energies_requested = value
        energies_undelivered = energies_undelivered_for_each_day[key]
        plt.bar([i for i in range(len(energies_requested))], energies_requested, color='lightblue', edgecolor='lightblue')
        plt.bar([i for i in range(len(energies_requested))], energies_undelivered, color='black', edgecolor='black')

        # Add titles and labels
        # plt.title('Sample Bar Chart', fontsize=16)
        plt.xlabel('Nabíjanie elektromobily', fontsize=12)
        plt.ylabel('Energia (kWh)', fontsize=12)

        # Add gridlines
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        num_bars = len(energies_requested)
        tick_positions = list(range(0, num_bars, tick_after_bars))  # Positions for labels (19, 39, 59, ...)
        custom_labels = [f'{tick_after_bars*i}' for i in range(len(tick_positions))]

        # Set ticks and labels
        plt.xticks(ticks=tick_positions, labels=custom_labels)

        # Add a legend with comments only
        plt.legend(['dodaná energia', 'nedodaná energia'], fontsize='medium')
        # Show the plot
        plt.show()


# this type of input data is not used in RL algorithm, because of the lack of data of this type
def load_json_ev_data(document,
                      start:datetime,
                      end:datetime,
                      period=5,
                      amount_of_evs_interval=None,
                      max_charging_rate_within_interval=None):
    f = open(document)
    data = json.load(f)
    evs = []
    evs_time_not_normalised = []
    if amount_of_evs_interval is None:
        num_of_evs = np.inf
    else:
        num_of_evs = random.randint(
            amount_of_evs_interval[0],
            amount_of_evs_interval[1])
    ev_index = 0
    for ev_dict in data["_items"]:
        arrival_datetime = datetime.strptime(
            ev_dict["connectionTime"],
            '%a, %d %b %Y %H:%M:%S %Z')

        arrival_timestamp = datetime_to_timestamp(
            start=start,
            chosen_date=arrival_datetime,
            period=period)

        departure_datetime = datetime.strptime(
            ev_dict["disconnectTime"],
            '%a, %d %b %Y %H:%M:%S %Z')

        departure_timestamp = datetime_to_timestamp(
            start=start,
            chosen_date=departure_datetime,
            period=period)
        energy_requested = ev_dict["kWhDelivered"]

        if start <= arrival_datetime <= end and start <= departure_datetime <= end:
            # how is peak charging rate calculated in acnportal? it seems there is no data on that one
            # maximum_charging_rate = random.randint(1,4)
            # maximum_charging_rate = 2*round((energy_requested/(departure_timestamp - arrival_timestamp + 1)) + 0.001,3)

            maximum_charging_rate = ((2 * energy_requested) /
                                     (departure_timestamp - arrival_timestamp + 1))
            if max_charging_rate_within_interval is not None:
                maximum_charging_rate = random.uniform(
                    max_charging_rate_within_interval[0],
                    max_charging_rate_within_interval[1])
            add_index = []
            if amount_of_evs_interval is None:
                add_index = [ev_index]
            evs.append(add_index + [arrival_timestamp,
                        departure_timestamp,
                        maximum_charging_rate,
                        energy_requested])

            evs_time_not_normalised.append(
                add_index + [arrival_datetime,
                 departure_datetime,
                 maximum_charging_rate,
                 energy_requested])
            if amount_of_evs_interval is None:
                ev_index += 1


    return num_of_evs, evs, evs_time_not_normalised


# time of the day [0,24)
def random_choice_evs_dict(evs:list,
                           evs_time_not_normalised:list,
                           num_of_evs):
    if num_of_evs == np.inf:
        return evs, evs_time_not_normalised
    chosen_idxs = []
    randomly_chosen_evs = {}
    randomly_chosen_evs_not_normalised_time = {}
    while len(randomly_chosen_evs) < num_of_evs:
        index = random.randint(0, len(evs_time_not_normalised) - 1)
        if index not in chosen_idxs:
            key, value = evs[index]
            chosen_idxs.append(index)
            if randomly_chosen_evs.get(key, None) is None:
                randomly_chosen_evs[key] = [value]
                randomly_chosen_evs_not_normalised_time[key] = [evs_time_not_normalised[key]]
            else:
                randomly_chosen_evs[key].append(value)
                randomly_chosen_evs_not_normalised_time[key].append(evs_time_not_normalised[key])
    return randomly_chosen_evs, randomly_chosen_evs_not_normalised_time




def random_choice_evs(evs:list,
                      evs_time_not_normalised:list,
                      num_of_evs):
    if num_of_evs == np.inf:
        return evs, evs_time_not_normalised
    chosen_idxs = []
    randomly_chosen_evs = []
    randomly_chosen_evs_not_normalised_time = []
    ev_index = 0
    while len(randomly_chosen_evs) < num_of_evs:
        index = random.randint(0, len(evs) - 1)
        if index not in chosen_idxs:
            chosen_idxs.append(index)
            randomly_chosen_evs.append([ev_index] + evs[index])
            randomly_chosen_evs_not_normalised_time.append([ev_index] + evs_time_not_normalised[index])
            ev_index += 1
    return randomly_chosen_evs, randomly_chosen_evs_not_normalised_time


def get_evs_data_from_document_advanced_settings(
        document,
        start:datetime,
        end:datetime,
        number_of_evs_interval,
        period=5,
        allow_overday_charging=True,
        include_weekends=False,
        max_charging_rate_within_interval=None,
        dates_in_ios_format=False


):
    date_to_evs_dict_timestamp_reseted = {}
    date_to_evs_dict_timestamp_not_reseted = {}
    date_to_evs_dict_time_not_normalised = {}
    with open(document, 'r') as file:
        data = json.load(file)
        for ev in data["_items"]:
            connection_time = None
            disconnect_time = None
            if dates_in_ios_format:
                connection_time = datetime.fromisoformat(ev['connectionTime'])
                disconnect_time = datetime.fromisoformat(ev['disconnectTime'])
            else:
                date_format = '%a, %d %b %Y %H:%M:%S GMT'
                connection_time = datetime.strptime(ev['connectionTime'], date_format)
                disconnect_time = datetime.strptime(ev['disconnectTime'], date_format)

            la_lt = pytz.timezone('America/Los_Angeles')
            connection_time = connection_time.astimezone(la_lt)
            disconnect_time = disconnect_time.astimezone(la_lt)
            start_of_day = connection_time.replace(hour=0, minute=0, second=0, microsecond=0)


            energy_requested = ev["kWhDelivered"]
            maximum_charging_rate = 6.6
            if max_charging_rate_within_interval is not None:
                maximum_charging_rate = random.uniform(max_charging_rate_within_interval[0],
                                                       max_charging_rate_within_interval[1])
            if not include_weekends and is_weekend(connection_time):
                continue
            if not allow_overday_charging and connection_time.date() != disconnect_time.date():
                continue
            if not (start <= connection_time <= end):
                continue
            if not (connection_time <= disconnect_time):
                continue
            if not (energy_requested >= 0):
                continue
            arrival_timestamp_reseted = datetime_to_timestamp(start=start_of_day, chosen_date=connection_time, period=period)
            departure_timestamp_reseted = datetime_to_timestamp(start=start_of_day, chosen_date=disconnect_time,
                                                              period=period)
            arrival_timestamp_not_reseted = datetime_to_timestamp(start=start, chosen_date=connection_time,
                                                              period=period)
            departure_timestamp_not_reseted = datetime_to_timestamp(start=start, chosen_date=disconnect_time,
                                                                period=period)

            ev_index = len(date_to_evs_dict_timestamp_reseted.get(start_of_day, []))
            ev_timestamp_reseted = [
                ev_index,
                arrival_timestamp_reseted,
                departure_timestamp_reseted,
                maximum_charging_rate,
                energy_requested
            ]
            ev_timestamp_not_reseted = [
                ev_index,
                arrival_timestamp_not_reseted,
                departure_timestamp_not_reseted,
                maximum_charging_rate,
                energy_requested
            ]
            ev_time_not_normalised = [
                ev_index,
                connection_time,
                disconnect_time,
                maximum_charging_rate,
                energy_requested

            ]

            if date_to_evs_dict_timestamp_reseted.get(start_of_day, None) is None:
                date_to_evs_dict_timestamp_reseted[start_of_day] = []
                date_to_evs_dict_timestamp_not_reseted[start_of_day] = []
                date_to_evs_dict_time_not_normalised[start_of_day] = []
            date_to_evs_dict_timestamp_reseted[start_of_day].append(ev_timestamp_reseted)
            date_to_evs_dict_timestamp_not_reseted[start_of_day].append(ev_timestamp_not_reseted)
            date_to_evs_dict_time_not_normalised[start_of_day].append(ev_time_not_normalised)
        copy_of_dict = copy.deepcopy(date_to_evs_dict_timestamp_reseted)
        for key, value in copy_of_dict.items():
            if not (number_of_evs_interval[0] <= len(value) <= number_of_evs_interval[1]):
                del date_to_evs_dict_timestamp_reseted[key]
                del date_to_evs_dict_timestamp_not_reseted[key]
                del date_to_evs_dict_time_not_normalised[key]
                continue
    return (date_to_evs_dict_timestamp_reseted,
            date_to_evs_dict_timestamp_not_reseted,
            date_to_evs_dict_time_not_normalised)


def plot_hourly_arrivals(evs_time_not_normalised, title=''):
    arrivals_in_time = np.zeros(shape=(24,))
    for key, value in evs_time_not_normalised.items():
        for v in value:
            ev_index, connection_time,disconnect_time,maximum_charging_rate,energy_requested = v
            arrivals_in_time[connection_time.hour] += 1
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(0, 24, 1), arrivals_in_time, marker='o', linestyle='-', color='b')

    plt.title(title)
    # Adding labels and title
    # plt.title('Mean Squared Error per Day')
    plt.xlabel('Hodiny')
    plt.ylabel('Počet prichodov elektromobilov')
    # plt.xticks(x_values)  # Set x ticks to be the indices
    plt.xticks(np.arange(0, 24, 1))
    plt.grid()
    plt.legend()

    plt.ylim(bottom=0)
    plt.xlim(left=0, right=24)
    # Show the plot
    plt.show()
def plot_hourly_requested_energy(evs_time_not_normalised, title=''):
    requested_energy_in_time = np.zeros(shape=(24,))
    for key, value in evs_time_not_normalised.items():
        for v in value:
            ev_index, connection_time, disconnect_time, maximum_charging_rate, energy_requested = v
            for t in range(24):
                if connection_time.hour <= t <= disconnect_time.hour:
                    requested_energy_in_time[t] += energy_requested
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(0, 24, 1), requested_energy_in_time, marker='o', linestyle='-', color='b')

    plt.title(title)
    # Adding labels and title
    # plt.title('Mean Squared Error per Day')
    plt.xlabel('Hodiny')
    plt.ylabel('Množstvo požadovanej energie')
    # plt.xticks(x_values)  # Set x ticks to be the indices
    plt.xticks(np.arange(0, 24, 1))
    plt.grid()
    plt.legend()

    plt.ylim(bottom=0)
    plt.xlim(left=0, right=24)
    # Show the plot
    plt.show()

def plot_hourly_departures(evs_time_not_normalised, title=''):
    departures_in_time = np.zeros(shape=(24,))
    for key, value in evs_time_not_normalised.items():
        for v in value:
            ev_index, connection_time, disconnect_time, maximum_charging_rate, energy_requested = v
            departures_in_time[disconnect_time.hour] += 1
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(0, 24, 1), departures_in_time, marker='o', linestyle='-', color='b')

    plt.title(title)
    # Adding labels and title
    # plt.title('Mean Squared Error per Day')
    plt.xlabel('Hodiny')
    plt.ylabel('Počet odchod elektromobilov')
    # plt.xticks(x_values)  # Set x ticks to be the indices
    plt.xticks(np.arange(0, 24, 1))
    plt.grid()
    plt.legend()

    plt.ylim(bottom=0)
    plt.xlim(left=0, right=24)
    # Show the plot
    plt.show()


# caltech's website for ev data is not accessible, therefore we don't often use this function for extracting input data
# but rather we access data through load_time_series_ev_data especially in experiments involving RL
def get_evs_data_from_document(
    document,
    start:datetime,
    end:datetime,
    period=5,
    amount_of_evs_interval=None,
    # in acnportal they use default values for maximum charging rates unless explicitly changed
    max_charging_rate_within_interval=None

    # max_battery_power,
    # force_feasible=False,
):
    '''

    Args:

        document: json document containing charging sessions
        start: start date of ev arrivals/departures
        end: end date of ev departures/arrivals
        period: length of interval between consecutive timesteps
        amount_of_evs_interval:
        max_charging_rate_within_interval:


    Returns:

    '''
    num_of_evs, evs, evs_time_not_normalised = load_json_ev_data(document=document,
                      start=start,
                      end=end,
                      period=period,
                      amount_of_evs_interval=amount_of_evs_interval)
    randomly_choose_n_evs, randomly_chosen_evs_not_normalised_time = random_choice_evs(evs=evs,evs_time_not_normalised=evs_time_not_normalised,num_of_evs=num_of_evs)
    return randomly_choose_n_evs, randomly_chosen_evs_not_normalised_time






def create_table(charging_profiles_matrix,
                 charging_cost_vector,
                 period,
                 capacity_in_time,
                 show_charging_costs=False,
                 path_to_save=''):
    '''
    Creates table of charging rates for a whole day
    Args:
        charging_profiles_matrix:
        charging_cost_vector:
        period: time between consecutive timesteps
        show_charging_costs: the variable which specifies if charging costs are also included in table

    Returns:

    '''
    timesteps_per_hour = (60//period)

    data = {}
    rows = [f'{i + 1}.auto' for i in range(len(charging_profiles_matrix))] + ['Celkové množstvo nabitej energie']
    data['čas (v hodinách)'] = [f'{i + 1}.auto' for i in range(len(charging_profiles_matrix))]
    data['čas (v hodinách)'].append('Súčet nabitej energie v čase')
    data['čas (v hodinách)'].append('Kapacita v čase')
    if show_charging_costs:
        data['čas (v hodinách)'].append('Náklady spotrebitelov za nabijanie energie v čase')
    start_hour = 0
    end_hour = 24
    overall_costs_for_evs_charging_per_hour = []
    overall_costs_for_energy_that_is_allocated_for_ev_charging = []
    values = np.zeros(shape=(len(charging_profiles_matrix) + 1,len(list(range(start_hour, end_hour))) + 1))
    for i in range(start_hour, end_hour):
        # charging rates and prices for 1 hour 0:5, ...
        charging_rates = charging_profiles_matrix[:, (i * timesteps_per_hour):((i + 1) * timesteps_per_hour)]
        current_prices = charging_cost_vector[(i * timesteps_per_hour):((i + 1) * timesteps_per_hour)]
        # data[f'{i + 1}'] = list(np.around(np.sum(charging_rates, axis=1),decimals=3)) + [round(np.sum(charging_rates,axis=None),3)]
        capacities = capacity_in_time[(i * timesteps_per_hour):((i + 1) * timesteps_per_hour)]
        if show_charging_costs:
            charging_ev_cost = (charging_rates) @ current_prices
            energy_allocated_costs = [capacities[i]*current_prices[i] for i in range(len(current_prices))]
            data[f'{i + 1}'] = (list(np.sum(charging_rates, axis=1)) +
                                [np.sum(charging_rates, axis=None)] +
                                [math.fsum(capacities)] +
                                [math.fsum(charging_ev_cost)])
            overall_costs_for_evs_charging_per_hour.append(math.fsum(charging_ev_cost))
            overall_costs_for_energy_that_is_allocated_for_ev_charging.append(math.fsum(energy_allocated_costs))
        else:
            data[f'{i + 1}'] = (list(np.sum(charging_rates, axis=1)) +
                                [np.sum(charging_rates, axis=None)] +
                                [math.fsum(capacities)])
        # values[:,i] = list(np.around(np.sum(charging_rates, axis=1),decimals=3)) + [round(np.sum(charging_rates,axis=None),3)]
        values[:, i] = (list(np.sum(charging_rates, axis=1)) +
                        [np.sum(charging_rates, axis=None)])
    # tu pri vypocte cien zmenit kwh na kw
    overall_charged_energy_per_ev = list(np.sum(charging_profiles_matrix, axis=1))
    # fixnut este
    if show_charging_costs:
        data[f'{start_hour}-{end_hour}'] = (overall_charged_energy_per_ev +
                                            [sum(overall_charged_energy_per_ev)] +
                                            [sum(capacity_in_time)]+
                                            [math.fsum(overall_costs_for_evs_charging_per_hour)])
        data[f'({start_hour}-{end_hour})'] = (list((charging_profiles_matrix) @
                                                  charging_cost_vector.T) +
                                              [math.fsum(overall_costs_for_evs_charging_per_hour)] + [math.fsum(overall_costs_for_energy_that_is_allocated_for_ev_charging)] + ['-'])
    else:
        data[f'{start_hour}-{end_hour}'] = (overall_charged_energy_per_ev +
                                            [math.fsum(overall_charged_energy_per_ev)]+
                                            [math.fsum(capacity_in_time)])
    # values[:,-1] = overall_charged_energy_per_ev + [sum(overall_charged_energy_per_ev)]


    df = pd.DataFrame(data)
    print(df)
    if len(path_to_save) == 0:
        df.to_csv('results.csv')
    else:
        df.to_csv(path_to_save)

def get_hourly_ut(uts, period):
    steps = (60//period)
    res = []
    for i in range(1,24+1):
        res.append(np.mean(uts[steps*(i-1):steps*i]))
    return np.array(res)
# evs = [[arrival,departure,requested_energy], ...]
def save_evs_to_file(filename,
                     evs,
                     evs_with_time_not_normalised,
                     timestamped_time= False,
                     with_labels=False,
                     set_maximum_charging_rate=None,
                     period=12):
    # Open the file for writing
    all_evs = []
    all_evs_time_not_normalised = []

    if isinstance(evs, dict) and isinstance(evs_with_time_not_normalised, dict):
        for key, value in evs.items():
             all_evs += value
        for key, value in evs_with_time_not_normalised.items():
            all_evs_time_not_normalised += value
    else:
        all_evs = copy.deepcopy(evs)
        all_evs_time_not_normalised = copy.deepcopy(evs_with_time_not_normalised)

    with open(filename, 'w') as f:
        f.write('format i.auto: a_i, d_i, r_i, e_i\n')
        # f.write('format i.auto: a_i, d_i, r_i\n')
        for i, ev in enumerate(all_evs, start=1):
            index, arrival, departure, maximum_charging_rate, requested_energy = ev
            index, arrival_not_normalised, departure_not_normalised, maximum_charging_rate, requested_energy = all_evs_time_not_normalised[i - 1]
            # requested_energy/= (period/60)
            if set_maximum_charging_rate is not None:
                maximum_charging_rate = set_maximum_charging_rate
            # maximum_charging_rate /= (period/60)
            if timestamped_time is False:
                arrival_not_normalised_minutes_str = str(
                    arrival_not_normalised.minute)
                departure_not_normalised_minutes_str = str(
                    departure_not_normalised.minute)
                if len(arrival_not_normalised_minutes_str) < 2:
                    arrival_not_normalised_minutes_str = '0' + arrival_not_normalised_minutes_str
                if len(departure_not_normalised_minutes_str) < 2:
                    departure_not_normalised_minutes_str = '0' + departure_not_normalised_minutes_str

                if with_labels is False:
                    # f.write(f'auto {i}: {arrival_not_normalised.hour}:{arrival_not_normalised_minutes_str}, '
                    #         f'{departure_not_normalised.hour}:{departure_not_normalised_minutes_str}, {maximum_charging_rate} kW, {requested_energy} kW\n')
                    f.write(f'{i}.auto: {arrival_not_normalised.hour}:{arrival_not_normalised_minutes_str}, '
                            f'{departure_not_normalised.hour}:{departure_not_normalised_minutes_str}, {maximum_charging_rate} kW, {requested_energy} kWh\n')
                if with_labels:
                    # f.write(f'auto {i}:  a_{i} = {arrival_not_normalised.hour}:{arrival_not_normalised_minutes_str}, '
                    #         f'd_{i} = {departure_not_normalised.hour}:{departure_not_normalised_minutes_str}, r_{i} = {maximum_charging_rate} kW, e_{i} = {requested_energy}\n')
                    f.write(f'{i}.auto :  a_{i} = {arrival_not_normalised.hour}:{arrival_not_normalised_minutes_str}, '
                            f'd_{i} = {departure_not_normalised.hour}:{departure_not_normalised_minutes_str}, r_{i} = {maximum_charging_rate} kW\n')
            else:
                f.write(f'{i}.auto:  a_{i} = {arrival}, '
                        f'd_{i} = {departure}, r_{i} = {maximum_charging_rate} kW\n')


def convert_dict_evs_to_list_evs(evs:dict):
    res = []
    for key, value in evs.items():
        res += value
    return res

def bisection_search_Pt(EVs,
                        start,
                        end,
                        algorithm,
                        period=5,
                        error_tol=1e-8,
                        number_of_evse=54,
                        cost_function=None,
                        costs_loaded_manually=None):
    lower_bound_Pt = 0
    upper_bound_Pt = sum([ev[-1] for ev in EVs])
    while abs(upper_bound_Pt - lower_bound_Pt) >= error_tol:
        middle = (lower_bound_Pt + upper_bound_Pt) / 2

        scheduling_alg = algorithm(EVs=EVs,
                             start=start,
                             end=end,
                             available_energy_for_each_timestep=middle,
                             time_between_timesteps=period,
                             accuracy=1e-8,
                             number_of_evse=number_of_evse,
                             cost_function=cost_function,
                             process_output=True,
                             costs_loaded_manually=costs_loaded_manually)

        try:
            is_solution_feasible, charging_rates = scheduling_alg.solve()

        except SolverError:
            is_solution_feasible, charging_rates = False, None


        if is_solution_feasible:
            upper_bound_Pt = middle
        else:
            lower_bound_Pt = middle
    # we should use upper bound to make sure we have enough energy
    return upper_bound_Pt
def undelivered_energy_file_rl(filename, evs_to_undelivered_dict):
    with open(filename, 'w') as f:
        f.write('auto i: nedodana energia, pozadovana energia \n')
        total_undelivered_energy = 0
        total_requested_energy = 0
        for key,value in evs_to_undelivered_dict.items():
            requested_energy, undelivered_energy = value
            if undelivered_energy < 0:
                undelivered_energy = 0
            total_undelivered_energy += undelivered_energy
            total_requested_energy += requested_energy

            f.write(f'auto {key}:{undelivered_energy} kWh, {requested_energy} kWh \n')
        f.write(f'Celkova nedodana energia {total_undelivered_energy} kWh \n')
        f.write(f'Celkova pozadovana energia {total_requested_energy} kWh')
def write_xt_states_into_file(filename,xts):
    with open(filename, 'w') as f:
        for i, xt in enumerate(xts, start=0):
            f.write(f't = {i} \n')
            f.write(f'{str(xt)}\n \n')


def write_into_file_operator_optimisation(filename, pts, generated_uts, costs_per_u, results, beta, U):
    with open(filename,'w') as f:
        f.write(f'Format: \n')
        f.write('casovy krok t \n'
                'mnozina U \n '+
                'flexibilita pt \n' +
                'ceny jednotlivych ut \n' +
                'beta  \n'+
                'hodnoty ct(ut) - B*log pt pre kazde u \n'+
                'vybrate (vyhladene) ut \n \n'
                )
        for i,pt in enumerate(pts, start=0):
            f.write(f't = {i} \n')
            f.write(f'U = {str(U)} \n \n')
            f.write(f'pt = {str(pt)}\n')
            f.write(f'c(uts) = {str(costs_per_u[i])}\n')
            f.write(f'beta = {str(beta)}\n')
            f.write(f'ct(ut) - B*log pt pre kazde ut = {results[i]} \n')
            f.write(f'vybrate (vyhladene) ut {generated_uts[i]} \n \n')



def undelivered_energy_file(filename, evs, charging_plan):

    with open(filename, 'w') as f:
        f.write('auto i: nedodana energia, pozadovana energia \n')
        total_undelivered_energy = 0
        total_requested_energy = 0
        for ev in evs:

            index, arrival_time,departure_time, maximum_charging_rate, requested_energy = ev
            undelivered_energy = math.fsum(charging_plan[index]) - requested_energy
            if undelivered_energy < 0:
                undelivered_energy = 0
            total_undelivered_energy += undelivered_energy
            total_requested_energy += requested_energy

            f.write(f'auto {index}:{undelivered_energy} kWh, {requested_energy} kWh \n')
        f.write(f'Celkova nedodana energia {total_undelivered_energy} kWh\n')
        f.write(f'Celkova pozadovana energia {total_requested_energy} kWh')
def write_evaluation_rewards_into_file(filename, charging_days_list, rewards):
    with open(filename, 'w') as f:
        f.write('Testovaci den = priemerne odmeny za dany den (epizodu)\n')
        for i,rew in enumerate(rewards, start=0):
            f.write(f'{charging_days_list[i]} = {rew} \n')


def create_settings_file(filename,
                         evs_num:int,
                         start:datetime,
                         end:datetime,
                         time_horizon:list,
                         number_of_evse:int,
                         period:int,
                         algorithm_name:str,
                         charging_networks_chosen:list,
                         garages_chosen:list,
                         cost_function:str='t',
                         operational_constraint=150,
                         manually_computed_costs_hourly = None,
                         solver_name:str='SCIP',
                         set_of_signals=None):
    with open(filename, 'w') as f:
        f.write(f'Pocet aut = {evs_num}\n')
        f.write(f'zaciatok nabijania = {start}\n')
        f.write(f'koniec nabijania = {end}\n')
        f.write(f'casova zona =  Amerika/Los Angeles\n')
        f.write(f'Dlzka casoveho horizontu T = {len(time_horizon)}\n')
        if manually_computed_costs_hourly is None:
            f.write(f'Cenova funkcia: c_nt = {cost_function}\n')
        else:
            ...
            # f.write(f'Ceny za 1kW po hodinach {manually_computed_costs_hourly}\n')
        f.write(f'Pocet nabijaciek = {number_of_evse}\n')
        charging_networks_str = ', '.join(charging_networks_chosen)
        garages_str = ', '.join(garages_chosen)
        f.write(f'Data o autach sme ziskali z nabijacich stanic {charging_networks_str} a z ich garazi {garages_str}\n')
        # f.write(f'')
        # f.write(f'P(t) = {available_energy_for_each_timestep}\n')
        f.write(f'Operacne obmedzenia = ut <= {operational_constraint}kW\n')
        f.write(f'Cas medzi susednymi casovymi krokmi = {period} min\n')
        f.write(f'Pouzity algoritmus = {algorithm_name}\n')
        f.write(f'Pouzity Offline optimal solver = {solver_name}\n')
        if set_of_signals is None:
            set_u = np.linspace(0,150,10)
            f.write(f'Mnozina uskutocnitelnych akcii U = {set_u}\n')
        else:
            f.write(f'Mnozina uskutocnitelnych akcii U = {set_of_signals}\n')
def mpe_for_more_days(schedule_for_each_day, evs_for_each_day):
    overall_energy_delivered = 0
    overall_energy_requested = 0

    for day, ev in enumerate(evs_for_each_day, start=0):
        index, arrival_time, departure_time, maximum_charging_rate, requested_energy = ev
        overall_energy_requested += requested_energy
        # overall_energy_delivered += math.fsum(schedule[index,:])
    return overall_energy_delivered / overall_energy_requested


def mse_error_fun_rl_testing(sum_of_charging_rates, ut_signals, capacity_constraint,period=12):
    res_mse_error = 0
    for col in range(len(sum_of_charging_rates)):
        res_mse_error += abs(sum_of_charging_rates[col] - ut_signals[col])**2 / (capacity_constraint* (period/60))
    return res_mse_error
# for given gamma or beta
def calculate_mpe_from_charging_rates_over_all_days(charging_rates, evs_for_each_day, charging_days_list):
    overall_energy_delivered = 0
    overall_energy_requested = 0
    for k in range(len(charging_rates)):
        specific_day = charging_days_list[k]
        evs_for_specific_day = evs_for_each_day[specific_day]
        for i in range(len(evs_for_specific_day)):
            index,arrival,departure, maximum_charging_rate,energy_requested = evs_for_specific_day[i]
            overall_energy_requested += energy_requested
            for t in range(len(charging_rates[k][i])):
                overall_energy_delivered += charging_rates[k][i][t]
    return 1 - (overall_energy_delivered/ overall_energy_requested)




def mpe_error_fun_rl_testing(ev_diction):
    overall_energy_requested = 0
    overall_energy_delivered = 0
    for ev, ev_values in ev_diction.items():
        requested_energy, undelivered_energy = ev_values
        ev_delivered_energy = requested_energy - undelivered_energy
        overall_energy_requested += requested_energy
        overall_energy_delivered += ev_delivered_energy
    return 1 - (overall_energy_delivered / overall_energy_requested)

def mpe_error_fun(schedule, evs):
    if len(evs) == 0:
        return 1
    overall_energy_delivered = 0
    overall_energy_requested = 0
    for ev in evs:
        index, arrival_time, departure_time, maximum_charging_rate, requested_energy = ev
        overall_energy_requested += requested_energy
        overall_energy_delivered += math.fsum(schedule[index,:])
    return 1 - (overall_energy_delivered / overall_energy_requested)
def write_results_into_file(filename, charging_rates, price_vector):
    with open(filename, 'w') as f:
        f.write(...)

def find_number_of_decimal_places(number):
    str_num = str(number)
    split_int, split_decimal = str_num.split('.')
    return len(split_decimal)
# day ahead market for one day
def load_locational_marginal_prices(filename,organization, period):
    if not filename.endswith('.csv'):
        raise ValueError('Filename must end with .csv')
    num_of_steps = int(60/period)
    hours = []
    prices_hourly = []
    prices = np.zeros(shape=(int((24*60)/period)))
    # Load and iterate over the CSV file
    with open(f'{filename}', mode='r') as file:
        csv_reader = csv.reader(file)

        # Skip the header if there is one
        header = next(csv_reader)
        i = 0
        average = 0
        num = 0
        for index, row in enumerate(csv_reader):
            # row is list of strings created by separting by commas
            date, price, given_organisation = row
            price = float(price)
            if given_organisation == organization:
                prices[i*num_of_steps:(i+1)*num_of_steps] = price
                prices_hourly.append(price)
                i += 1
            elif organization is None:
                if len(hours) > 0 and date == hours[-1]:
                    num += 1
                    average += price
                    continue
                if len(hours) > 0 and date != hours[-1]:
                    average /= num
                    prices[i * num_of_steps:(i + 1) * num_of_steps] = average
                    prices_hourly.append(average)
                    i += 1
                    num = 1
                    average = price
                    hours.append(date)
                    continue

                hours.append(date)
                average += price
                num += 1

        if organization is None:
            average /= num
            prices_hourly.append(average)
            prices[i * num_of_steps:(i + 1) * num_of_steps] = average
            hours.append(date)

        return prices_hourly, prices
#
def load_locational_marginal_prices_per_year(filenames, period, organisation=None):
    possible_organisations = ['PGAE', 'SCE', 'SDGE', 'VEA']
    average_lmps_hourly = np.zeros(shape=(24,))
    num_of_organisations = 4
    start_of_year = datetime(2016, 1, 1)
    start_of_next_year = datetime(2017, 1, 1)

    # Calculate the difference in days
    days_in_year = (start_of_next_year - start_of_year).days
    last_hour = 0
    for i in range(len(filenames)):
        row_num = 0
        with open(f'{filenames[i]}', mode='r') as file:
            csv_reader = csv.reader(file)
            for index, row in enumerate(csv_reader):
                if row_num == 0:
                    row_num += 1
                    continue
                date, price, given_organisation = row
                if organisation is not None and given_organisation != organisation:
                    continue
                price = float(price)
                # Define the format that matches the input string
                date_format = '%m/%d/%Y %I:%M:%S %p'

                # Convert the string to datetime
                date_obj = datetime.strptime(date, date_format)
                average_lmps_hourly[date_obj.hour] += price
                row_num += 1
    if organisation is None:
        average_lmps_hourly /= num_of_organisations
    average_lmps_hourly /= days_in_year
    average_lmps_per_timestep = []
    for i in range(int((24*60)/period)):
        index = int(i // (60 / period))
        average_lmps_per_timestep.append(average_lmps_hourly[index])
    return average_lmps_hourly, average_lmps_per_timestep
def default_settings_json_file():
    # learning starts parameter we can possibly change
    # action space shouldnt be a problem looking at relu activation function
    json_file = {
        'optimizer':'Adam',
        # learning rate matches with sb3
        'learning rate': 3e-4,
        # replay buffer size matches with sb3
        'replay buffer size': 10^6,
        'number of hidden layers (all networks)':2,
        'number of hidden units per layer': 256,

        'number of samples per minibatch': 256,
        'Non-linearity':'ReLU',
        # this parameter gamma is different in sb3 - need to change it for sb3
        'discount factor': 0.5,
        # target smoothing coefficient matches with sb3
        'target smoothing coefficient':5e-3,
        # gradient step matches with sb3
        'gradient steps': 1,
        # target update interval matches with sb3
        'target update interval':1,

        # add more info if RL environment does more things like smoothing
        'number of evse': 54,
        'number of power levels': 10,
        'tuning parameter': 6e3,
        'number of episodes': 500,
        'scheduling algorithm':'LLF',
        # entrophy coeff - we set it to learn automatically, but we can try 0.5 but then reward scaling issues can occur
        'temperature parameter': 0.5,
        'Power rating':150,
        'Reward':'H(pt) + o1*sum_{i in active evs}||s_t(i)||_2'
                 '- o2* sum_{i in active evs} I(t = d(i))[e_{t}(j)]'
                 '- o3* |sum(s_{t}) - PPC(pt,ct)|',
        'Maximum charging rate for each EV':6.6,
        'time interval':12,
        'o1':0.1,
        'o2':0.2,
        'o3':2}
    return json_file

def save_hyperparam_into_file(filename,input_diction):
    default_dict = default_settings_json_file()
    for key, value in input_diction.items():
        if key in default_dict:
            default_dict[key] = value
    save_to_json(default_dict,filename)

def save_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def set_list(value, num_evs):
    res_list = []
    if isinstance(value, list):
        if len(value) == num_evs:
            raise ValueError('Bad number of arrival times')
        res_list = copy.deepcopy(value)
    elif isinstance(value, int) or isinstance(value, float):
        res_list = [value for _ in range(num_evs)]
    else:
        raise ValueError('Bad type of arrival timestamps')
    return res_list
#         for one day for testing
def create_dataset(arrival_timestamps,
                   departure_timestamps,
                   maximum_charging_rates,
                   requested_energies,
                   num_evs):
    res_arrival_timestamps = set_list(value=arrival_timestamps,num_evs=num_evs)
    res_departure_timestamps = set_list(value=departure_timestamps,num_evs=num_evs)
    res_maximum_charging_rates = set_list(value=maximum_charging_rates,num_evs=num_evs)
    res_requested_energies = set_list(value=requested_energies,num_evs=num_evs)

    evs = []
    for i in range(num_evs):
        arrival_timestamp = res_arrival_timestamps[i]
        departure_timestamp = res_departure_timestamps[i]
        maximum_charging_rate = res_maximum_charging_rates[i]
        requested_energy = res_requested_energies[i]
        ev = [i, arrival_timestamp, departure_timestamp, maximum_charging_rate,requested_energy]
        evs.append(ev)
    return evs

def convert_timestep_to_hours(timestep, time_between_timesteps):
    return (timestep*time_between_timesteps) / 60

def save_data_to_json_via_acn_api(start,
                          end,
                          site,
                          path_to_file_save,
                          token="DEMO_TOKEN",
                          min_kwh=None
                          ):
    client = DataClient(token)
    docs = client.get_sessions_by_time(site, start, end)
    data_diction = {"meta":{}, "_items":[]}

    data_diction["meta"]["start"] = datetime.isoformat(start)
    data_diction["meta"]["end"] = datetime.isoformat(end)
    data_diction["meta"]["site"] = site
    data_diction["meta"]["min_kWh"] = min_kwh
    for d in docs:

        d['connectionTime'] = datetime.isoformat(d['connectionTime'])
        d['disconnectTime'] = datetime.isoformat(d['disconnectTime'])
        if d['doneChargingTime'] is not None:
            d['doneChargingTime'] = datetime.isoformat(d['doneChargingTime'])

        data_diction["_items"].append(d)

    with open(path_to_file_save, "w") as json_file:
        json.dump(data_diction, json_file, indent=4)

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

def dummy_expert(env,_obs):
    """
    Random agent. It samples actions randomly
    from the action space of the environment.

    :param _obs: (np.ndarray) Current observation
    :return: (np.ndarray) action taken by the expert
    """
    return env.action_space.sample()


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record("random_value", value)
        return True
