import math
import copy
import numpy as np
# in OPT solver decides whether solution is feasible or not
# used in other algorithms
from testing_functions import (check_charging_rates_within_bounds,
                               check_infrastructure_not_violated,
                               check_all_energy_demands_met)
def is_solution_feasible(EVs,
                         charging_rates,
                         power_limit,
                         algorithm_name,
                         gamma = 1,
                         period=12,
                         algorithm_precision=1e-6,
                         num_of_evse=54):
    if not check_charging_rates_within_bounds(evs=EVs, charging_rates=charging_rates):
        return False
    if not check_infrastructure_not_violated(charging_rates=charging_rates,
                                             power_limit=power_limit,
                                             period=period,
                                             accuracy=algorithm_precision):
        return False
    # problems with accuracy
    if not check_all_energy_demands_met(evs=EVs,
                                        charging_rates=charging_rates,
                                        gamma=gamma,
                                        algorithm_name=algorithm_name):
        return False
    # if not are_enough_evses(schedule=charging_rates,num_of_evse=num_of_evse):
    #     return False
    return True
def are_enough_evses(evs, num_of_evse):
    for ev_index in range(len(evs)):
        arr = [ev_index]
        for ev2_index in range(ev_index + 1, len(evs)):
            index1, arrival_time1, departure_time1, maximum_charging_rate1, requested_energy1 = evs[ev_index]
            index2, arrival_time2, departure_time2, maximum_charging_rate2, requested_energy2 = evs[ev2_index]
            if departure_time2 >= arrival_time1 >= arrival_time2:
                arr.append(ev2_index)
            if departure_time2 >= departure_time1 >= arrival_time2:
                arr.append(ev2_index)
        if len(arr) > num_of_evse:
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






