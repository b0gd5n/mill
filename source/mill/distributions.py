
class Random:
    def __init__(self, generator, minimal_value=-0.5, maximal_value=0.5):
        self.generator = generator
        self.minimal_value = minimal_value
        self.maximal_value = maximal_value

    def generate(self, inputs, outputs):
        weights = []

        for weight in range(inputs * outputs):
            weights.append(self.generator.generate(minimal_value=self.minimal_value, maximal_value=self.maximal_value))

        return weights
