import math
import numpy as np

def test_flexibility(pt):
    return math.fsum(pt) == 1

def check_charging_rates_within_bounds(evs, charging_rates):
    for ev_index in range(len(charging_rates)):
        index, ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = evs[ev_index]
        for time_index in range(len(charging_rates[ev_index])):
            # diff = abs(charging_rates[i][j] - evs[i][maximum_charging_rate_index])
            if time_index > ev_departure or time_index < ev_arrival:
                if charging_rates[ev_index][time_index] != 0:
                    return False
            else:
                if charging_rates[ev_index][time_index] < 0 or charging_rates[ev_index][time_index] > ev_maximum_charging_rate:
                    return False
    return True

# possibly if ut is time varying change the function, but for now it is enough
def check_infrastructure_not_violated(charging_rates,
                                      power_limit,
                                      period,
                                      accuracy=1e-6):
    for col in range(charging_rates.shape[1]):
        if math.fsum(charging_rates[:, col]) <= power_limit[col] * (period / 60):
            continue
        if abs(math.fsum(charging_rates[:, col]) - power_limit[col]*(period/60)) > accuracy:
            return False
    return True

def check_all_energy_demands_met(evs,
                                 charging_rates,
                                 algorithm_name,
                                 gamma = 1,
                                 algorithm_accuracy=1e-6):
    # problems with multiplication of decimal numbers - inaccuracies
    if gamma != 1:
        return True
    for ev_index in range(len(charging_rates)):
        index, ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = evs[ev_index]
        accuracy = find_number_of_decimal_places(ev_requested_energy)
        # there can be multiple times the same inaccuracies, so lowering accuracy by 1 decimal works
        if math.fsum(charging_rates[ev_index]) == gamma*ev_requested_energy:
            continue
        # precision loss when multiplying by beta
        if (math.fsum(charging_rates[ev_index]) - gamma*ev_requested_energy) > algorithm_accuracy:
            return False
        # if (round(math.fsum(charging_rates[ev_index]), min(accuracy, algorithm_accuracy_decimals)) - round(gamma*ev_requested_energy, algorithm_accuracy_decimals)) > algorithm_accuracy_decimals:
        #     return False

    return True

def check_number_of_taken_evse(charging_rates, number_of_evse=54):
    if charging_rates.ndim == 1:
        if np.count_nonzero(charging_rates) > number_of_evse:
            return False
    elif charging_rates.ndim == 2:
        for i in range(charging_rates.shape[0]):
            if np.count_nonzero(charging_rates[:,i]) > number_of_evse:
                return False
    return True




def find_number_of_decimal_places(number):
    if isinstance(number, int):
        return 0
    str_num = str(number)
    split_int, split_decimal = str_num.split('.')
    return len(split_decimal)
