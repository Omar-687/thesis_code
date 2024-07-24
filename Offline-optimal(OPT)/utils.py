import math
import random
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
from numpy import number

from opt_algorithm import OPT
import matplotlib.pyplot as plt


def datetime_to_timestamp(start, chosen_date, period, round_up=False):
    """ Convert a datetime object to a timestamp measured in simulation periods.

    Args:
        dt (datetime): Datetime to be converted to a simulation timestamp.
        period (int): Length of one time interval in the simulation. (minutes)
        round_up (bool): If True, round up when casting timestamp to int, else round down.

    Returns:
        int: dt expressed as a simulation timestamp.
    """
    ts = (chosen_date - start) / timedelta(minutes=period)
    if round_up:
        return int(math.ceil(ts))
    else:
        return int(ts)

# time of the day [0,24)

def convert_to_evs(
    document,
    start:datetime,
    end:datetime,
    period=5,
    time_horizon_length=288,
    amount_of_evs_interval=[5,10],
    # in acnportal they use default values for maximum charging rates unless explicitly changed
    max_charging_rate_within_interval=[1,4]

    # max_battery_power,
    # force_feasible=False,
):
    '''

    Args:
        document: json document containing charging sessions
        start: start date of ev arrivals/departures
        end: end date of ev departures/arrivals
        period: length of interval between consecutive timesteps

    Returns:

    '''

    f = open(document)
    data = json.load(f)
    evs = []
    evs_time_not_normalised = []

    if amount_of_evs_interval is None:
        n = np.inf
    else:
        n = random.randint(amount_of_evs_interval[0], amount_of_evs_interval[1])
    for ev_dict in data["_items"]:
        arrival_datetime = datetime.strptime(ev_dict["connectionTime"], '%a, %d %b %Y %H:%M:%S %Z')
        arrival_timestamp = datetime_to_timestamp(start=start,chosen_date=arrival_datetime, period=period)
        departure_datetime = datetime.strptime(ev_dict["disconnectTime"], '%a, %d %b %Y %H:%M:%S %Z')
        departure_timestamp = datetime_to_timestamp(start=start, chosen_date=departure_datetime,period=period)
        # not all data have requested energy rounded to 3 decimal places
        # energy_requested = round(ev_dict["kWhDelivered"], 3)
        energy_requested = ev_dict["kWhDelivered"]

        if start <= arrival_datetime <= end and start <= departure_datetime <= end:
            # how is peak charging rate calculated in acnportal? it seems there is no data on that one
            # should i use simulator or not needed at all?
            # maximum_charging_rate = random.randint(1,4)
            # maximum_charging_rate = 2*round((energy_requested/(departure_timestamp - arrival_timestamp + 1)) + 0.001,3)
            maximum_charging_rate =  (2 *energy_requested) / (departure_timestamp - arrival_timestamp + 1)
            evs.append([arrival_timestamp, departure_timestamp, maximum_charging_rate, energy_requested])
            evs_time_not_normalised.append([arrival_datetime, departure_datetime, maximum_charging_rate, energy_requested])

    if n == np.inf:
        return evs, evs_time_not_normalised
    chosen_idxs = []
    randomly_chosen_evs = []
    randomly_chosen_evs_not_normalised_time = []
    while len(randomly_chosen_evs) < n:
        index = random.randint(0, len(evs_time_not_normalised) - 1)
        if index not in chosen_idxs:
            chosen_idxs.append(index)
            randomly_chosen_evs.append(evs[index])
            randomly_chosen_evs_not_normalised_time.append(evs_time_not_normalised[index])
        if len(chosen_idxs) >= len(evs):
            break



    # chosen_evs = random.sample(evs,max_amount_of_evs)
    return randomly_chosen_evs, randomly_chosen_evs_not_normalised_time

def create_table(charging_profiles_matrix,
                 charging_cost_vector,
                 period,
                 # P_t,
                 update_after_hours,
                 # start:datetime,
                 # end:datetime,
                 show_charging_costs=False):
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
            data[f'{i + 1}'] = list(np.sum(charging_rates, axis=1)) + [np.sum(charging_rates, axis=None)] + [math.fsum(charging_ev_cost)]
            overall_costs_for_evs_charging_per_hour.append(math.fsum(charging_ev_cost))
        else:
            data[f'{i + 1}'] = list(np.sum(charging_rates, axis=1)) + [np.sum(charging_rates, axis=None)]
        # values[:,i] = list(np.around(np.sum(charging_rates, axis=1),decimals=3)) + [round(np.sum(charging_rates,axis=None),3)]
        values[:, i] = list(np.sum(charging_rates, axis=1)) + [np.sum(charging_rates, axis=None)]
    overall_charged_energy_per_ev = list(np.sum(charging_profiles_matrix, axis=1))
    if show_charging_costs:
        data[f'{start_hour}-{end_hour}'] = overall_charged_energy_per_ev + [sum(overall_charged_energy_per_ev)] + [math.fsum(overall_costs_for_evs_charging_per_hour)]
        data[f'({start_hour}-{end_hour})'] = list(charging_profiles_matrix @ charging_cost_vector.T) + [math.fsum(overall_costs_for_evs_charging_per_hour)] + ['-']
    else:
        data[f'{start_hour}-{end_hour}'] = overall_charged_energy_per_ev + [sum(overall_charged_energy_per_ev)]
    # values[:,-1] = overall_charged_energy_per_ev + [sum(overall_charged_energy_per_ev)]


    df = pd.DataFrame(data)
    print(df)
    df.to_csv('results.csv')

    # fig = plt.figure(figsize=(8, 2))
    # ax = fig.add_subplot(111)
    #
    # ax.table(cellText=df.values,
    #          rowLabels=rows,
    #          colLabels=list(df.keys()),
    #          loc="center"
    #          )
    # ax.set_title("Top 10 Fields of Research by Aggregated Funding Amount")
    #
    # ax.axis("off")
    #
    # plt.savefig('ev_charging_results.png')




# evs = [[arrival,departure,requested_energy], ...]
def save_evs_to_file(filename, evs:list, evs_with_time_not_normalised:list, timestamped_time= False, with_labels=False):
    # Open the file for writing
    with open(filename, 'w') as f:
        f.write('format i.auto: a_i, d_i, r_i, e_i\n')
        # f.write('format i.auto: a_i, d_i, r_i\n')
        for i, ev in enumerate(evs, start=1):
            arrival, departure, maximum_charging_rate, requested_energy = ev
            arrival_not_normalised, departure_not_normalised, maximum_charging_rate, requested_energy = evs_with_time_not_normalised[i - 1]
            if timestamped_time is False:
                arrival_not_normalised_minutes_str = str(arrival_not_normalised.minute)
                departure_not_normalised_minutes_str = str(departure_not_normalised.minute)
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
                f.write(f'auto {i}:  a_{i} = {arrival}, '
                        f'd_{i} = {departure}, r_{i} = {maximum_charging_rate} kW\n')
# finding  minimal P_t

def bisection_search_Pt_offline(evs,
                                period=5,
                                error_tol=1e-4,
                                num_of_days=1):
    lb_Pt = 0
    ub_Pt = sum([ev[-1] for ev in evs])
    while abs(ub_Pt - lb_Pt) >= error_tol:
        middle = (lb_Pt + ub_Pt) / 2

        scheduling_alg = OPT(EVs=evs,
                             P_t=middle,
                             time_between_timesteps=period,
                             num_of_days=num_of_days,
                             price_function=None)

        charging_rates = scheduling_alg.solve()
        # energy_charged = np.sum(charging_rates, axis=1)
        requested_energy_index = 3
        # correct_solution = [ev[requested_energy_index] for ev in evs] == [ev for ev in energy_charged]

        if charging_rates is not None:
            ub_Pt = middle
        else:
            lb_Pt = middle



    return ub_Pt

def create_settings_file(filename,
                         evs_num:int,
                         start:datetime,
                         end:datetime,
                         T,
                         P_t,
                         period:int,
                         alg_name:str,
                         charging_station:str='caltech',
                         cost_function:str='t',
                         solver_name:str='ECOS'):
    with open(filename, 'w') as f:
        f.write(f'Počet áut = {evs_num}\n')
        f.write(f'zaciatok nabíjania = {start}\n')
        f.write(f'koniec nabíjania = {end}\n')
        f.write(f'Dĺžka časového horizontu T = {len(T)}\n')
        f.write(f'Cenová funkcia: c_nt = {cost_function}\n')
        # f.write(f'')
        f.write(f'P(t) = {P_t}\n')
        f.write(f'Čas medzi susednými časovými krokmi = {period} min\n')
        f.write(f'Použitý algoritmus = {alg_name}\n')
        f.write(f'Použitý LP solver = {solver_name}')
def write_results_into_file(filename, charging_rates, price_vector):
    with open(filename, 'w') as f:
        f.write(...)

def find_number_of_decimal_places(number):
    str_num = str(number)
    split_int, split_decimal = str_num.split('.')
    return len(split_decimal)