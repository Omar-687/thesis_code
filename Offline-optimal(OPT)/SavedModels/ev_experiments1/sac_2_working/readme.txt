works quite well, but doesnt correctly match signal ut
mpe is very similar to the one in article


 mse = self.o3*abs(self.signal_ut - math.fsum(schedule))**2/self.power_rating
        # mse = self.o3 * abs(self.signal_ut - math.fsum(schedule))
        # we include in schedule only currently charged evs
        reward = (entropy(action) + self.o1*math.fsum(schedule)
                  - self.o2 * not_fully_charged_until_departure_penalty - mse)
        # third part is important else there is not much to learn if ut is totally random
        # the agent will not learn if ut is random bc it will satisfy its problems anyway for many different solutions
        reward = float(reward) / 100


           o1=0.1,
                 o2=30,
                 o3=1

   changes in reward radical - possibly the penalty is too high, but still it learned
   but after more episodes - like 150 k min
   smoothing = 0.35

   sac 23 i think