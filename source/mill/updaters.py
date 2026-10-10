class StochasticGradientDescent:
    def __init__(self, parameters):
        self.parameters = parameters

    def step(self, rate):
        for parameter in range(len(self.parameters)):
            for value in range(len(self.parameters[parameter].values)):
                self.parameters[parameter].values[value] -= rate * self.parameters[parameter].gradients[value]

    def clear(self):
        for parameter in range(len(self.parameters)):
            self.parameters[parameter].clear()
