mpe is a bit worse than theirs but maybe it doesnt matter that much
mse seems ok
try with different beta

100 k krokov trenovany


def __init__(self,
                 scheduling_algorithm,
                 charging_days_list,
                 cost_list,
                 max_charging_rate,
                 tuning_parameter,
                 train=True,
                 evse=54,
                 power_levels=10,
                 time_between_timesteps=12,
                 power_limit=150,
                 training_episodes=500,
                 limit_ramp_rates=True,
                 o1=0.1,
                 o2=0.2,
                 # o2=0.2,
                 # o1=0.1,
                 # o2=0.2,
                 o3=2,
                 costs_in_kwh=False
                 # o3=0.2
                 ):
                 same other parameters as for 20.11 last experiment
                 averaging of 2 values probably as smoothing