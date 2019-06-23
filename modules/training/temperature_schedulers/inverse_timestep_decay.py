from overrides import overrides
from modules.training.temperature_schedulers.temperature_scheduler import TemperatureScheduler


class InverseTimestepDecay(TemperatureScheduler):
    def __init__(self, t_initial, decay_rate):
        super().__init__(t_initial)
        self._time_step = 0
        self.decay_rate = decay_rate
    
    @overrides
    def step(self):
        self._time_step += 1
        self._temperature = self._temperature / (1 + self.decay_rate * self._time_step) 

