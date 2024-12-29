quite ok results

but probably lowering coefficients would be better because i think it negatively impacts
exploration

dividing all numbers by 3 and penalty for overcharge by 10
raise maximum charging rate = 3 so offline optimal can solve it for gamma = 1
change signal buffer to length = 5
graph for 16.12 looks very similar to one in publication
increae smoothing a bit to 0.8

decrease smoothing to decrease mse to 0.5


(self,
                 scheduling_algorithm,
                 charging_days_list,
                 cost_list,
                 train=True,
                 evse=54,
                 tuning_parameter=6e3,
                 max_charging_rate=2,
                 power_levels=10,
                 time_between_timesteps=12,
                 power_rating=150,
                 training_episodes=500,
                 limit_ramp_rates=True,
                 o1=15,
                 o2=40,
                 # o2=0.2,
                 # o1=0.1,
                 # o2=0.2,
                 o3=16
                 # o3=0.2
                 ):