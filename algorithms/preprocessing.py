
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