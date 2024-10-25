import math
import copy
from copy import deepcopy
import numpy as np
# in OPT solver decides whether solution is feasible or not
# used in other algorithms
from testing_functions import (check_charging_rates_within_bounds,
                               check_infrastructure_not_violated,
                               check_all_energy_demands_met)
def is_solution_feasible(EVs,
                         charging_rates,
                         available_energy_for_each_timestep,
                         algorithm_name):
    if not check_charging_rates_within_bounds(evs=EVs, charging_rates=charging_rates):
        return False
    if not check_infrastructure_not_violated(charging_rates=charging_rates,
                                             available_energy_for_each_timestep=available_energy_for_each_timestep):
        return False
    if not check_all_energy_demands_met(evs=EVs,
                                        charging_rates=charging_rates,
                                        algorithm_name=algorithm_name):
        return False
    return True
# works for one column schedule
def get_maximum_possible_charging_values_given_schedule(
        active_evs,
        schedule_shape,
        maximum_charging_rates_matrix,
        timestep):
    res = np.zeros(shape=schedule_shape[0])
    for i in range(len(active_evs)):
        index, arrival_time, departure_time, maximum_charging_rate, requested_energy = active_evs[i]
        # res[index, 0] = maximum_charging_rates_matrix[index, timestep]
        res[index] = maximum_charging_rates_matrix[index, timestep]
    return res


def find_number_of_decimal_places(number):
    str_num = str(number)
    split_int, split_decimal = str_num.split('.')
    return len(split_decimal)

def correct_charging_rate(charging_rate,
                          ev_remaining_energy_to_be_charged,
                          maximum_charging_rate):
    if charging_rate < 0:
        return 0
    elif charging_rate > min(ev_remaining_energy_to_be_charged,
                             maximum_charging_rate):
        return min(ev_remaining_energy_to_be_charged,
                   maximum_charging_rate)
    return charging_rate

def correct_charging_rates_offline(EVs,
                                   charging_rates,
                                   maximum_charging_rates_matrix):
    res = copy.deepcopy(charging_rates)
    for i in range(len(charging_rates)):
        index, ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = EVs[i]
        for j in range(len(charging_rates[i])):
            charging_rate = charging_rates[i][j]
            res[i][j] = correct_charging_rate(charging_rate=charging_rate,
                                  ev_remaining_energy_to_be_charged=ev_requested_energy,
                                  maximum_charging_rate=maximum_charging_rates_matrix[i][j])
            ev_requested_energy -= res[i][j]
    return res






