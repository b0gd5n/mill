class StochasticGradientDescent:
    def __init__(self, model_parameters):
        self.model_parameters = model_parameters

    def update_parameters(self, learning_rate):
        for parameter in range(len(self.model_parameters)):
            for value in range(len(self.model_parameters[parameter].values)):
                self.model_parameters[parameter].values[value] -= learning_rate * self.model_parameters[parameter].gradients[value]

    def clear_gradients(self):
        for parameter in range(len(self.model_parameters)):
            self.model_parameters[parameter].clear_gradients()
