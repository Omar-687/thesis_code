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
                 o1=0.2,
                 o2=0.3,
                 # o2=0.2,
                 # o1=0.1,
                 # o2=0.2,
                 o3=2,
                 costs_in_kwh=False
                 # o3=0.2
                 ):

                 everything else is same as previous experiment
                 problem with mse still mpe is not that huge problem
                 lower o1 a o2
                 try to change variables to kwh in reward