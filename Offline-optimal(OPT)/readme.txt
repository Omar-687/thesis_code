(
source zachary lee's phd thesis and acnsim article
and sLLF article)
laxity = remaining charging time - (remaining energy to be charged / maximum charging rate)

Class of scheduling algorithms - index policies.
 In index based policies, jobs are first sorted by a given index, then processed in that order.
 The primary objective of these index policies is to maximize throughput. One of seconday objectives is the fairness in charging.

least-laxity-first (LLF) (online algorithm)
input: active (connected) EVs
(as list of tuples (arrival, departure, requested energy, maximum charging rate)

1. sorting active EVs by given metric (we sort by laxity in increasing order)
2. processing them in such order
3. each EV is assigned the maximum feasible charging rate
(calculated by bisection alg, given that assignments to all previous EVs are fixed)

In this context, a feasible charging rate is one that does
 not cause an infrastructure constraint to be violated and is less than the maximum
 charging rate

feasible charging rate cant violate infrastructure constraints capacity
feasible charging rate < maximum charging rate

LLF alg:
input: all_EVs (represented in tuples a_i,d_i,r_i,e_i), (optimisation horizon depends on implementation)
schedule = np.zeros(shape = (len(evs), len(T)))
for t in T:
    active_EVs = get_active_evs(all_EVs)
    available_power_at_time_t = number (P_t)
    sorted_active_evs = sort(active_EVs, key=laxity)
    for ev in sorted_active_evs:
        maximum_feasible_charging_rate = bisection(maximum_charging_rate,
                                                   available_power_at_time_t,
                                                   remaining_requested_energy_at_time_t)

        schedule[ev_index][t] = maximum_feasible_charging_rate
        available_power_at_time_t -= maximum_feasible_charging_rate
    update_evs_state(active_EVs) (update the remaining requested energy only)
output: schedule

LLF bisection:
input: maximum_charging_rate, available_power_at_time_t, remaining_requested_energy_at_time_t
error_tolerance = 1e-4
lower_bound = 0
upper_bound = min(maximum_charging_rate, remaining_requested_energy_at_time_t)
while abs(upper_bound - lower_bound) > error_tolerance:
    maximum_feasible_charging_rate = (upper_bound + lower_bound) / 2
    if maximum_feasible_charging_rate > available_power_at_time_t:
        upper_bound = maximum_feasible_charging_rate
    else:
        lower_bound = maximum_feasible_charging_rate
output: lower_bound


source: Smoothed Least-Laxity-First Algorithm for EV
 Charging

Additionally, unlike the classic LLF algorithm, the sLLF
 algorithm avoids unnecessary oscillations in the charging rates

 optimisation problem which solution maximizes the minimum laxity at time t+1


smoothed-least-laxity-first (sLLF) (online algorithm)
input: all_EVs (represented in tuples a_i,d_i,r_i,e_i)
for t in T:
    active_EVs = get_active_evs(all_evs)
    optimal_L_t = bisection(active_EVs, available_power_at_time_t)
    optimal_charging_rates[:, t] = get_schedule(L_t, active_EVs, ...) (other parameters are those needed for solving equation to get optimal charging rate for each EV)
    update_evs_state(active_EVs) (update the remaining requested energy only)
output: optimal_charging_rates

bisection SLLF:
input: active_EVs, available_power_at_time_t
ev_laxities = get_laxities(active_EVs)
upper_bound_L_t = max(ev_laxities) (the result of operations in the bracket for every ev must be at least 1)
lower_bound_L_t = min(ev_laxities) - 1 (the result of operations in the bracket for every ev must be at least 1)
while (upper_bound_L_t - lower_bound_L_t) > error_tolerance:
   middle_L_t = (upper_bound_L_t - lower_bound_L_t) / 2
   if sum([maximum_charging_rate *
            (middle_L_t - ev_laxities[i] + 1) for i in active_EVs]) > available_power_at_time_t :
        upper_bound_L_t = middle_L_t
   else:
        lower_bound_L_t = middle_L_t
output: lower_bound_L_t

get_schedule
input:active_EVs, L_t, ev_laxities
schedule = []
for ev in active_EVs:
    ev_charging_rate = maximum_charging_rate * (L_t - ev_laxities[i] + 1)
    schedule.append(ev_charging_rate)
return schedule


EVSE assignment:
connecting to EVSE:
An EV i connects to an EVSE in time between [a_i,d_i]
(not necessarily at time a_i becase there might be no free EVSE)
disconnecting from EVSE:
The EV disconnects from the EVSE in time d_i or when all requested energy to the EV
was given


Number of EVSE:
if all EVs connected to EVSE at time of their arrival, increasing the number of EVSE
wont influence the solution
if there is no possibility to find feasible schedule for given number of EVSE, increase
the number of EVSE
