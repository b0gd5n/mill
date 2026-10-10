class LinearCongruentialGenerator:
    def __init__(self, seed=42):
        self.seed = seed
        self.state = seed % 4294967296

        self.usages = 0

    def generate(self, minimal_value, maximal_value):
        self.state = (1664525 * self.state + 1013904223) % 4294967296

        self.usages += 1

        return minimal_value + (maximal_value - minimal_value) * self.state / 4294967296
