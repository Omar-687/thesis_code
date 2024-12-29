because beta is 6e3
sacrifice mpe for mse a bit

max charging rate set to = 2
increase mse error it needs to have a bit bigger mpe
decrease thecharging rates further to 1.5


def __init__(self,
                 scheduling_algorithm,
                 charging_days_list,
                 cost_list,
                 train=True,
                 evse=54,
                 tuning_parameter=6e3,
                 max_charging_rate=6.6,
                 power_levels=10,
                 time_between_timesteps=12,
                 power_rating=150,
                 training_episodes=500,
                 limit_ramp_rates=True,
                 o1=9,
                 o2=40,
                 # o2=0.2,
                 # o1=0.1,
                 # o2=0.2,
                 o3=11
                 # o3=0.2
                 ):