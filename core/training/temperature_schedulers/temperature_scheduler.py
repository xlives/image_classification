class TemperatureScheduler:
    def __init__(self, t_intial):
        self._temperature = t_intial

    def step(self):
        raise NotImplementedError

    def get_temperature(self):
        return self._temperature
