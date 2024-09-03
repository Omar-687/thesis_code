number of EVSE - for now there assumption that everytime a new EV comes, it connects immediately to EVSE
(dont know how to do it differently because OPT doesnt take into account number of EVSE)

laxity = remaining time - (remaining energy to be charged / maximum charging rate)

least-laxity-first (LLF)
input: all EVs (as list of tuples (arrival, departure, requested energy, maximum charging rate)
1. sorting active EVs by given metric (in this case laxity)
2. processing them in such order
3. each EV is assigned the maximum feasible charging rate (how to calculate it? find ref)
(it is calculated by bisection alg,, given that assignments to all previous EVs are fixed)

bisection:
optimisation variable r_i
min_value = 0
max_value = min(P_t, maximum charging rate of EV i)


return charging plan of all EVs over whole time horizon


how to test it:


smoothed-least-laxity-first (sLLF)
