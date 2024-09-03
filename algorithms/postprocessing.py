import math

# in OPT solver decides whether solution is feasible or not
# used in other algorithms
def is_solution_feasible(EVs, charging_rates):
    feasible = True
    for i, ev_charging_plan in enumerate(charging_rates):
        ev_energy_delivered = math.fsum(ev_charging_plan)
        arrival, departure, maximum_charging_rate, requested_energy = EVs[i]
        accuracy = find_number_of_decimal_places(requested_energy)
        if round(ev_energy_delivered, accuracy) != requested_energy:
            feasible = False
    return feasible


def find_number_of_decimal_places(number):
    str_num = str(number)
    split_int, split_decimal = str_num.split('.')
    return len(split_decimal)