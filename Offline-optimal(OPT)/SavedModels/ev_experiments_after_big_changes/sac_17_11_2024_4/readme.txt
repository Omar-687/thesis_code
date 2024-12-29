class EVenvironment(gymnasium.Env):
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
                 o1=1.3,
                 o2=1,
                 # o2=0.2,
                 # o1=0.1,
                 # o2=0.2,
                 o3=2,
                 costs_in_mwh=False
                 # o3=0.2
                 ):

                 quite significant mse, possibly problem si that there are less options to choose suitable signal
                 without smoothing or smth - which would make next uts closer to each other - more available options