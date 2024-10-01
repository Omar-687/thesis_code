from datetime import datetime


# we use index before each EV in these functions
# EV i is active when current_timestep is in interval [a_i, d_i]
def get_active_evs(evs, current_timestep):
    active_evs = []
    for ev in evs:
        index, ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = ev
        if ev_arrival <= current_timestep <= ev_departure:
            active_evs.append(ev)
    return active_evs

def get_laxity_of_ev(ev, ev_remaining_energy_to_be_charged, current_timestep):
    index, ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = ev
    laxity = (ev_departure - current_timestep) - (ev_remaining_energy_to_be_charged / ev_maximum_charging_rate)
    return laxity

def is_ev_valid(ev,
                start=None,
                end=None):
    ev_arrival, ev_departure, ev_maximum_charging_rate, ev_requested_energy = ev
    if ev_arrival > ev_departure:
        return False
    if isinstance(start, datetime) and isinstance(end, datetime) and isinstance(ev_arrival, datetime) and isinstance(ev_departure, datetime):
        if start <= ev_arrival <= ev_departure <= end:
            ...
        else:
            return False
    if isinstance(ev_arrival, int) and isinstance(ev_departure, int):
        if ev_arrival < 0 or ev_departure < 0:
            return False
    if ev_maximum_charging_rate <= 0:
        return False
    if ev_requested_energy <= 0:
        return False
    return True
def are_dict_input_data_valid(evs:dict,
                              start=None,
                              end=None):
    for key, value in evs.items():
        start_date = key
        evs_in_same_day = value
        if not are_input_data_valid(evs=evs_in_same_day,
                                    start=start,
                                    end=end):
            return False
    return True
def are_input_data_valid(evs,
                         start=None,
                         end=None):
    for ev in evs:
        if not is_ev_valid(ev=ev,
                           start=start,
                           end=end):
            return False
    return True