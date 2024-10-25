
from datetime import datetime, timedelta, timezone

from opt_algorithm import OPT
from utils import *

import unittest
from os.path import exists
from testing_functions import (check_all_energy_demands_met,
                               check_infrastructure_not_violated,
                               check_charging_rates_within_bounds)

from preprocessing import (are_input_data_valid,
                           is_ev_valid,
                           are_dict_input_data_valid)

from scheduling_alg import SchedulingAlg

# preloading of training and testing data so that there is no need to load such big data more times

start_testing = datetime(2019, 12, 2, 0, 0, 0,tzinfo=timezone.utc)
        # end_testing = datetime(2020, 1, 1, 23, 59, 59,  tzinfo=timezone.utc)
end_testing = datetime(2019, 12, 2, 23, 59, 59, tzinfo=timezone.utc)
ut_interval = [0 , 150]
maximum_charging_rate = 6.6
number_of_evse = 54
period = 12
number_of_evs_interval = [30, np.inf]
train_evs_timestamp_reset, train_evs_timestamp_not_reset, train_evs_time_not_normalised_time_reset = load_time_series_ev_data(
    charging_network=charging_networks[0],
    garages=caltech_garages,
    start=start_testing,
    end=end_testing,
    period=period,
    max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
    number_of_evs_interval=number_of_evs_interval,
    include_weekends=False,
)





start_testing = datetime(2019, 12, 2, 0, 0, 0,tzinfo=timezone.utc)
        # end_testing = datetime(2020, 1, 1, 23, 59, 59,  tzinfo=timezone.utc)
end_testing = datetime(2019, 12, 2, 23, 59, 59, tzinfo=timezone.utc)
ut_interval = [0 , 150]
maximum_charging_rate = 6.6
number_of_evse = 54
period = 12
number_of_evs_interval = [30, np.inf]
test_evs_timestamp_reset, test_evs_timestamp_not_reset, test_evs_time_not_normalised_time_reset = load_time_series_ev_data(
    charging_network=charging_networks[0],
    garages=caltech_garages,
    start=start_testing,
    end=end_testing,
    period=period,
    max_charging_rate_within_interval=[maximum_charging_rate, maximum_charging_rate],
    number_of_evs_interval=number_of_evs_interval,
    include_weekends=False,
)
