class Parameter:
    def __init__(self, values, regularized=True):
        self.values = values
        self.regularized = regularized
        
        self.gradients = []

        for value in range(len(values)):
            self.gradients.append(0)

    def clear(self):
        for gradient in range(len(self.gradients)):
            self.gradients[gradient] = 0
