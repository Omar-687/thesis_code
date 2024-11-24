aby videla skolitelka ze ten algoritmus dobre minimalizuje ceny za nabijanie energie
tak musim zvysit limit na kapacitu pre jpl pre caltech moze zostat rovnaka
o1 needs to be lower because mse error still causes some issues


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
                 o1=0.4,
                 o2=1,
                 # o2=0.2,
                 # o1=0.1,
                 # o2=0.2,
                 o3=2,
                 costs_in_kwh=False
                 # o3=0.2
                 ):