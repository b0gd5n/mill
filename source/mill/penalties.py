class Lasso:
    def __init__(self, parameters, strength=1e-5):
        self.parameters = parameters
        self.strength = strength

    def apply(self):
        for parameter in range(len(self.parameters)):
            if self.parameters[parameter].regularized:
                for value in range(len(self.parameters[parameter].values)):
                    if self.parameters[parameter].values[value] > 0:
                        self.parameters[parameter].gradients[value] += self.strength
                    elif self.parameters[parameter].values[value] < 0:
                        self.parameters[parameter].gradients[value] -= self.strength

class Ridge:
    def __init__(self, parameters, strength=1e-4):
        self.parameters = parameters
        self.strength = strength

    def apply(self):
        for parameter in range(len(self.parameters)):
            if self.parameters[parameter].regularized:
                for value in range(len(self.parameters[parameter].values)):
                    self.parameters[parameter].gradients[value] += self.strength * self.parameters[parameter].values[value]
