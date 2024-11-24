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
                 o1=3.7,
                 o2=4,
                 # o2=0.2,
                 # o1=0.1,
                 # o2=0.2,
                 o3=8.5,
                 costs_in_mwh=False
                 # o3=0.2
                 ):
        self.number_of_additional_observation_parameters = 2
        self.scheduling_algorithm = scheduling_algorithm
        low_vector = np.zeros(shape=(evse*2+self.number_of_additional_observation_parameters,))
        high_vector = np.zeros(shape=(evse*2+self.number_of_additional_observation_parameters,))
        self.evse = evse
        self.aggregator_state = np.zeros(self.evse * 2 + self.number_of_additional_observation_parameters)
        self.dict_arrivals_departures = {}
        for i in range(self.evse):
            high_vector[i*2] = 24
            # lets say this value can be infinite but we have to set limit for observation
            high_vector[(i * 2) + 1] = 100
        high_vector[-1] = 24
        high_vector[-2] = 150
        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        # Box - Supports continuous (and discrete) vectors or matrices, used for vector observations

        self.observation_space = spaces.Box(low=low_vector, high=high_vector, dtype=np.float64)
        low_bound_action_space = np.zeros(shape=(power_levels,))
        upper_bound_action_space = np.ones(shape=(power_levels,))
        self.action_space = spaces.Box(low=low_bound_action_space, high=upper_bound_action_space)

        self.chosen_day_index = 0
        self.activated_evse = np.zeros(shape=(self.evse,))
        self.evse_map_to_ev = np.zeros(shape=(self.evse,)) - 1
        self.charging_days_list = charging_days_list
        self.power_levels = power_levels
        self.costs_list = cost_list
        self.time_between_timesteps = time_between_timesteps
        self.power_limit = power_limit
        self.delta = (self.time_between_timesteps/60)
        # self.signal_buffer = deque(maxlen=3)
        self.signal_buffer = deque(maxlen=5)
        self.max_charging_rate = max_charging_rate
        self.tuning_parameter = tuning_parameter
        self.charging_days = charging_days_list
        self.timestep = 0
        # not necessary because max charging rate is the same for all EVs
        # self.map_evse_to_ev = []
        self.limit_ramp_rates = limit_ramp_rates
        self.train = train
        self.o1 = o1
        self.o2 = o2
        self.o3 = o3
        # add signal to the observation space
        self.signal_ut = 0
        self.cumulative_costs = 0
        self.smoothing = True
        # possibly too small coefficient
        # self.smoothing_coeff = 0.35
        # self.smoothing_coeff = 0.5
        # self.smoothing_coeff = 0.45
        # self.smoothing_coeff = 0.35
        self.smoothing_coeff = 0.6
        self.chosen_ut_for_each_timestep = []
        self.chosen_sum_of_charging_rates = []
        self.charging_rates_matrix = np.array([])
        # tuples (index of ev, undelivered energy) delivered energy we will find in outside function
        self.delivered_and_undelivered_energy = {}
        self.costs_in_mwh = costs_in_mwh
 la_tz = pytz.timezone('America/Los_Angeles')
        start_testing = la_tz.localize(datetime(2018, 11, 1, 0, 0, 0))
        end_testing = la_tz.localize(datetime(2019, 12, 1, 23, 59, 59))
        # maximum_charging_rate = 6.6
        maximum_charging_rate = 7
        period = 12
        max_number_of_episodes = 500
        number_of_timesteps_in_one_episode = ((60 * 24) / period)
        total_timesteps = max_number_of_episodes * number_of_timesteps_in_one_episode
        number_of_evs_interval = [30, np.inf]
        # this data must be loaded even if environment loads data separately, to filter charging days
        evs_timestamp_reset, evs_timestamp_not_reset, evs_time_not_normalised_time_reset = load_time_series_ev_data(
            charging_network=charging_networks[0],
            # garages=caltech_garages,
            garages=caltech_garages,
            start=start_testing,
            end=end_testing,
            period=period,
            max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
            number_of_evs_interval=number_of_evs_interval,
            include_weekends=False,
            include_overday_charging=False

        )
        charging_days_list = list(evs_timestamp_reset.keys())
        # TODO: have forgotten to save the model
        # make environment faster
        # # haarnoja - even simpler tasks sometimes need 1000 000 steps to learn
        # total_timesteps = int(500000 / 5)
        total_timesteps = 100000
        # total_timesteps = 240000
        scheduling_algorithm = LeastLaxityFirstAlg
        beta = 6e3

        bez smoothingu

        third_term = self.o3 * abs(self.signal_ut - math.fsum(schedule_in_kw)) /10
        # mse = self.o3 * abs(self.signal_ut - math.fsum(schedule))
        # we include in schedule only currently charged evs
        reward = (entropy(action) + self.o1*math.fsum(schedule_in_kw)
                  - self.o2 * not_fully_charged_until_departure_penalty - third_term)