class Lasso:
    def __init__(self, model_parameters, strength=1e-5):
        self.model_parameters = model_parameters
        self.strength = strength

    def apply_penalty(self):
        for parameter in range(len(self.model_parameters)):
            if self.model_parameters[parameter].regularized:
                for value in range(len(self.model_parameters[parameter].values)):
                    if self.model_parameters[parameter].values[value] > 0:
                        self.model_parameters[parameter].gradients[value] += self.strength
                    elif self.model_parameters[parameter].values[value] < 0:
                        self.model_parameters[parameter].gradients[value] -= self.strength

class Ridge:
    def __init__(self, model_parameters, strength=1e-4):
        self.model_parameters = model_parameters
        self.strength = strength

    def apply_penalty(self):
        for parameter in range(len(self.model_parameters)):
            if self.model_parameters[parameter].regularized:
                for value in range(len(self.model_parameters[parameter].values)):
                    self.model_parameters[parameter].gradients[value] += self.strength * self.model_parameters[parameter].values[value]
