import copy
import math
import random
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cvxpy import SolverError
import gzip, csv
import os

from keras.src.metrics.accuracy_metrics import accuracy
from tensorflow.core.grappler import costs

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
                                       include_weekends=False):
    # only files with this format are accepted, and the further checking of gzipped files is elsewhere
    if not file_name.endswith('.csv.gz'):
        return False
    splitted_string = file_name.split('-')[4:-1]
    file_date_arr = splitted_string[:2] + splitted_string[2].split('T') + splitted_string[3:]

    year, month, day, hour, minute, second = file_date_arr
    file_date = datetime(int(year),
                         int(month),
                         int(day),
                         int(hour),
                         int(minute),
                         int(second),
                         tzinfo=timezone.utc)

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
                                         max_charging_rate_within_interval=None):

    lines = None
    try:
        lines = file.readlines()
    except (OSError, ValueError):
        return False, None, None

    # first line is notation
    # second line should be start of charging
    # last line is end of charging possibly
    if len(lines) < 3:
         return False, None, None
    energy_index = -2

    # Get the last and second last lines
    second_line_start_charging_date = datetime.fromisoformat(lines[1].split(',')[0])
    last_line_end_charging_date = datetime.fromisoformat(lines[-1].split(',')[0])
    second_last_line_delivered_energy = 0
    if lines[-2].split(',')[energy_index] != '':
        second_last_line_delivered_energy = float(lines[-2].split(',')[energy_index])

    # delete evs that have not charged anything
    if second_last_line_delivered_energy == 0:
        return False, None, None
    start_of_day = second_line_start_charging_date.replace(hour=0, minute=0, second=0, microsecond=0)
    arrival_time = second_line_start_charging_date
    departure_time = last_line_end_charging_date
    energy_requested = second_last_line_delivered_energy

    arrival_timestamp_reseted = datetime_to_timestamp(start=start_of_day, chosen_date=arrival_time, period=period)
    departure_timestamp_reseted = datetime_to_timestamp(start=start_of_day, chosen_date=departure_time, period=period)

    arrival_timestamp = datetime_to_timestamp(start=start, chosen_date=arrival_time,
                                              period=period)
    departure_timestamp = datetime_to_timestamp(start=start, chosen_date=departure_time,
                                                period=period)


    max_charging_rate = (2 * energy_requested) / (departure_timestamp - arrival_timestamp + 1)

    if max_charging_rate_within_interval is not None:
        max_charging_rate = random.uniform(max_charging_rate_within_interval[0],
                                           max_charging_rate_within_interval[1])
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

            for i, ev in enumerate(value,0):
                res_diction_timestamp_reseted[key].append([index_of_ev] + ev)
                res_diction_timestamp_not_reseted[key].append([index_of_ev] + date_to_evs_diction_timestamp_not_reseted[key][i])
                res_diction_time_not_normalised[key].append([index_of_ev] + date_to_evs_diction_time_not_normalised[key][i])
                index_of_ev += 1
    else:
        res_diction_timestamp_reseted = deepcopy(date_to_evs_diction_timestamp_reseted)
        res_diction_timestamp_not_reseted = deepcopy(date_to_evs_diction_timestamp_not_reseted)
        res_diction_time_not_normalised = deepcopy(date_to_evs_diction_time_not_normalised)
        # date_to_evs_diction.clear()
        # date_to_evs_diction_time_not_normalised.clear()
        # date_to_evs_diction = copy.deepcopy(res_diction)
        # date_to_evs_diction_time_not_normalised = copy.deepcopy(res_diction_time_not_normalised)

    # if amount_of_evs_interval is not None:
    #     all_evs = []
    #     all_evs_not_normalised = []
    #     for key in date_to_evs_diction.keys():
    #         for ev in date_to_evs_diction[key]:
    #             all_evs.append([key, ev])
    #         for ev in date_to_evs_diction_time_not_normalised[key]:
    #             all_evs_not_normalised.append([key, ev])
    #     num_of_evs = random.randint(amount_of_evs_interval[0],
    #                                 amount_of_evs_interval[1])
    #
    #     res_diction.clear()
    #     res_diction_time_not_normalised.clear()
    #
    #     res_diction, res_diction_time_not_normalised = random_choice_evs_dict(evs=all_evs,
    #                                                                           evs_time_not_normalised=all_evs_not_normalised,
    #                                                                           num_of_evs=num_of_evs)
    # if res_diction == {}:
    #     res_diction = copy.deepcopy(date_to_evs_diction)
    #     res_diction_time_not_normalised = copy.deepcopy(date_to_evs_diction_time_not_normalised)

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
                             include_weekends=False):


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

    check_package_exists('ACNDataStatic')
    # subdirectory = 't'
    with (importlib.resources.path('ACNDataStatic', '.') as package_path):
        package_dir = Path(package_path)

        # Construct the full path to the subdirectory you want to access

        for i in range(len(garages)):
            subdirectory = f'time series data/{charging_network}/{garages[i]}'
            subdirectory_path = package_dir / subdirectory
            # Check if the subdirectory exists
            if subdirectory_path.is_dir():
                # List all files in the subdirectory
                for item in subdirectory_path.iterdir():
                    file_name = item.name

                    if not check_time_series_file_correctness(
                            file_name=file_name,
                            start=start,
                            end=end,
                            include_weekends=include_weekends):
                        continue

                    if not is_gzipped(subdirectory_path/file_name):
                        continue

                    with gzip.open(subdirectory_path/file_name, 'rt') as file:
                        # Read all lines from the file


                        value  = read_and_extract_time_series_ev_info(
                            file=file,
                            start=start,
                            period=period,
                            max_charging_rate_within_interval=max_charging_rate_within_interval)
                        if len(value) != 3:
                            a = 0
                        valid, extracted_ev_info_reseted,extracted_ev_info_not_reseted  = value[0], value[1], value[2]


                        if valid is False:
                            continue

                        start_of_day, arrival_time, arrival_timestamp_reseted, departure_time, departure_timestamp_reseted, max_charging_rate, energy_requested = extracted_ev_info_reseted
                        _, _, arrival_timestamp, _, departure_timestamp, _, _ = extracted_ev_info_not_reseted

                        ev_reseted = [arrival_timestamp_reseted,
                              departure_timestamp_reseted,
                              max_charging_rate,
                              energy_requested]
                        ev_not_reseted = [arrival_timestamp_reseted,
                              departure_timestamp_reseted,
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
                        else:
                            date_to_evs_diction_timestamp_reseted[start_of_day].append(ev_reseted)
                            date_to_evs_diction_time_not_normalised[start_of_day].append(ev_time_not_normalised)
                            date_to_evs_diction_timestamp_not_reseted[start_of_day].append(ev_not_reseted)

            else:
                # print(f"Subdirectory '{subdirectory}' does not exist.")
                return ValueError(f"Subdirectory '{subdirectory}' does not exist.")
        res_diction_timestamp_reseted, res_diction_timestamp_not_reseted, res_diction_time_not_normalised = filter_evs(
            date_to_evs_diction_timestamp_reseted=date_to_evs_diction_timestamp_reseted,
            date_to_evs_diction_timestamp_not_reseted=date_to_evs_diction_timestamp_not_reseted,
            date_to_evs_diction_time_not_normalised=date_to_evs_diction_time_not_normalised,
            number_of_evs_interval=number_of_evs_interval)

        return res_diction_timestamp_reseted, res_diction_timestamp_not_reseted, res_diction_time_not_normalised

# seems the costs are cumulative but i dont understand why they have such value
def mpe_cost_graph(mpe_values,cost_values):
    x_values = range(len(mpe_values))

    # Plotting the MSE values
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, cost_values, marker='o', linestyle='-', color='b')

    # Adding labels and title
    # plt.title('Mean Squared Error per Day')
    plt.xlabel('MPE')
    # find slovak translation of MSE
    plt.ylabel('Cena')
    # plt.xticks(x_values)  # Set x ticks to be the indices
    plt.grid()
    plt.legend()

    plt.ylim(bottom=0)
    plt.xlim(left=0, right=1)

    # Show the plot
    plt.show()

def get_ut_signals_from_schedule(schedule:np.array):
    cols = schedule.shape[1]
    uts = []
    for i in range(cols):
        uts.append(math.fsum(schedule[:, i]))
    return uts
def comparison_pilot_signal_real_signal_graph():
    ...
# seems the costs are cumulative but i dont understand why they have such value
def charging_in_time_graph(ut_signals_offline,
                           period,
                           ut_signals_ppc=None,
                           ut_signals_mpc = None):
    # x_values = range(len(ut_signals_offline))
    x_values = np.arange(0, 24, period/60)
    # Plotting the MSE values
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, ut_signals_offline, marker='o', linestyle='-', color='b')

    # Adding labels and title
    # plt.title('Mean Squared Error per Day')
    plt.xlabel('Hodiny')
    # find slovak translation of MSE
    plt.ylabel('Sila (kW)')
    # plt.xticks(x_values)  # Set x ticks to be the indices
    plt.grid()
    plt.legend()

    plt.ylim(bottom=0)
    plt.xlim(left=0, right=24)
    # Show the plot
    plt.show()


def calculate_cumulative_costs(schedule:np.array,cost_vector):
    res = 0
    cols = schedule.shape[1]
    for i in range(cols):
        res += math.fsum(cost_vector[i] * schedule[:, i])
    return res

def calculate_cumulative_costs_given_ut(uts:np.array,cost_vector):
    res = 0
    for i, ut in enumerate(uts, start=0):
        res += uts * cost_vector[i]
    return res


def costs_per_day_graph(costs):
    x_values = range(len(costs))

    # Plotting the MSE values
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, costs, marker='o', linestyle='-', color='b')

    # Adding labels and title
    # plt.title('Mean Squared Error per Day')
    plt.xlabel('Dni')
    # find slovak translation of MSE
    plt.ylabel('Kumulatívna cena energie')
    plt.xticks(x_values)  # Set x ticks to be the indices
    plt.grid()
    plt.legend()

    plt.ylim(bottom=0)
    # Show the plot
    plt.show()
def mpe_per_day_graph(mpe_values):
    # Create an array of indices (x values)
    x_values = range(len(mpe_values))

    # Plotting the MSE values
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, mpe_values, marker='o', linestyle='-', color='b')

    # Adding labels and title
    # plt.title('Mean Squared Error per Day')
    plt.xlabel('Dni')
    # find slovak translation of MSE
    plt.ylabel('MPE')
    plt.xticks(x_values)  # Set x ticks to be the indices
    plt.grid()
    plt.legend()

    # Set y-axis limits from 0 to 1
    plt.ylim(0, 1)

    # Show the plot
    plt.show()
def mse_per_day_graph(mse_values):
    # Create an array of indices (x values)
    x_values = range(len(mse_values))

    # Plotting the MSE values
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, mse_values, marker='o', linestyle='-', color='b', label='MSE Values')

    # Adding labels and title
    # plt.title('Mean Squared Error per Day')
    plt.xlabel('Dni')
    # find slovak translation of MSE
    plt.ylabel('MSE')
    plt.xticks(x_values)  # Set x ticks to be the indices
    plt.grid()
    plt.legend()

    # Set y-axis limits from 0 to 1
    plt.ylim(0, 1)
    # Show the plot
    plt.show()
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
            except IndexError:
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
        plt.ylabel('Energia (kW)', fontsize=12)

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
                 show_charging_costs=False):
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
    if show_charging_costs:
        data['čas (v hodinách)'].append('Náklady spotrebiteľov energie v čase')
    start_hour = 0
    end_hour = 24
    overall_costs_for_evs_charging_per_hour = []
    values = np.zeros(shape=(len(charging_profiles_matrix) + 1,len(list(range(start_hour, end_hour))) + 1))
    for i in range(start_hour, end_hour):
        charging_rates = charging_profiles_matrix[:, (i * timesteps_per_hour):((i + 1) * timesteps_per_hour)]
        current_prices = charging_cost_vector[(i * timesteps_per_hour):((i + 1) * timesteps_per_hour)]
        # data[f'{i + 1}'] = list(np.around(np.sum(charging_rates, axis=1),decimals=3)) + [round(np.sum(charging_rates,axis=None),3)]

        if show_charging_costs:
            charging_ev_cost = charging_rates @ current_prices
            data[f'{i + 1}'] = (list(np.sum(charging_rates, axis=1)) +
                                [np.sum(charging_rates, axis=None)] +
                                [math.fsum(charging_ev_cost)])
            overall_costs_for_evs_charging_per_hour.append(math.fsum(charging_ev_cost))
        else:
            data[f'{i + 1}'] = (list(np.sum(charging_rates, axis=1)) +
                                [np.sum(charging_rates, axis=None)])
        # values[:,i] = list(np.around(np.sum(charging_rates, axis=1),decimals=3)) + [round(np.sum(charging_rates,axis=None),3)]
        values[:, i] = (list(np.sum(charging_rates, axis=1)) +
                        [np.sum(charging_rates, axis=None)])
    overall_charged_energy_per_ev = list(np.sum(charging_profiles_matrix, axis=1))
    if show_charging_costs:
        data[f'{start_hour}-{end_hour}'] = (overall_charged_energy_per_ev +
                                            [sum(overall_charged_energy_per_ev)] +
                                            [math.fsum(overall_costs_for_evs_charging_per_hour)])
        data[f'({start_hour}-{end_hour})'] = (list(charging_profiles_matrix @
                                                  charging_cost_vector.T) +
                                              [math.fsum(overall_costs_for_evs_charging_per_hour)] + ['-'])
    else:
        data[f'{start_hour}-{end_hour}'] = (overall_charged_energy_per_ev +
                                            [sum(overall_charged_energy_per_ev)])
    # values[:,-1] = overall_charged_energy_per_ev + [sum(overall_charged_energy_per_ev)]


    df = pd.DataFrame(data)
    print(df)
    df.to_csv('results.csv')



# evs = [[arrival,departure,requested_energy], ...]
def save_evs_to_file(filename, evs, evs_with_time_not_normalised, timestamped_time= False, with_labels=False):
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
                            f'{departure_not_normalised.hour}:{departure_not_normalised_minutes_str}, {maximum_charging_rate} kW, {requested_energy} kW\n')
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


# TODO: if the problem is feasible or not - rather not because sometimes we dont know if it is feasible
# TODO: if cost function not specified, write from what cost array consists of - rather not
# TODO: change names of variables so they are the same across the whole project

def create_settings_file(filename,
                         evs_num:int,
                         start:datetime,
                         end:datetime,
                         time_horizon:list,
                         available_energy_for_each_timestep,
                         number_of_evse:int,
                         period:int,
                         algorithm_name:str,
                         charging_networks_chosen:list,
                         garages_chosen:list,
                         cost_function:str='t',
                         solver_name:str='ECOS'):
    with open(filename, 'w') as f:
        f.write(f'Pocet aut = {evs_num}\n')
        f.write(f'zaciatok nabijania = {start}\n')
        f.write(f'koniec nabijania = {end}\n')
        f.write(f'Dlzka casoveho horizontu T = {len(time_horizon)}\n')
        f.write(f'Cenova funkcia: c_nt = {cost_function}\n')
        f.write(f'Pocet nabijaciek = {number_of_evse}\n')
        charging_networks_str = ', '.join(charging_networks_chosen)
        garages_str = ', '.join(garages_chosen)
        f.write(f'Data o autach sme ziskali z nabijacich stanic {charging_networks_str} a z ich garazi {garages_str}\n')
        # f.write(f'')
        f.write(f'P(t) = {available_energy_for_each_timestep}\n')
        f.write(f'Cas medzi susednymi casovymi krokmi = {period} min\n')
        f.write(f'Pouzity algoritmus = {algorithm_name}\n')
        if algorithm_name == 'sLLF' or algorithm_name == 'LLF':
            ...
        else:
            f.write(f'Pouzity LP solver = {solver_name}')

def mpe_for_more_days(schedule_for_each_day, evs_for_each_day):
    overall_energy_delivered = 0
    overall_energy_requested = 0

    for day, ev in enumerate(evs_for_each_day, start=0):
        index, arrival_time, departure_time, maximum_charging_rate, requested_energy = ev
        overall_energy_requested += requested_energy
        # overall_energy_delivered += math.fsum(schedule[index,:])
    return overall_energy_delivered / overall_energy_requested


def mse_error_fun(schedule, evs):
    ...

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