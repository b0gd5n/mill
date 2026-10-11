class Lasso:
    def __init__(self, model_parameters):
        self.model_parameters = model_parameters

    def apply_penalty(self, strength=1e-5):
        for parameter in range(len(self.model_parameters)):
            if self.model_parameters[parameter].regularized:
                for value in range(len(self.model_parameters[parameter].values)):
                    if self.model_parameters[parameter].values[value] > 0:
                        self.model_parameters[parameter].gradients[value] += strength
                    elif self.model_parameters[parameter].values[value] < 0:
                        self.model_parameters[parameter].gradients[value] -= strength

class Ridge:
    def __init__(self, model_parameters):
        self.model_parameters = model_parameters

    def apply_penalty(self, strength=1e-4):
        for parameter in range(len(self.model_parameters)):
            if self.model_parameters[parameter].regularized:
                for value in range(len(self.model_parameters[parameter].values)):
                    self.model_parameters[parameter].gradients[value] += strength * self.model_parameters[parameter].values[value]
