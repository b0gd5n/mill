
class Cosine:
    def __init__(self, minimal_rate, maximal_rate, warming_steps, total_steps, iterations=16):
        self.maximal_rate = maximal_rate
        self.minimal_rate = minimal_rate
        self.warming_steps = warming_steps
        self.total_steps = total_steps
        self.iterations = iterations

    def cosine(self, angle):
        summand = 1
        term = 1

        for index in range(1, self.iterations):
            term *= -angle * angle / ((2 * index - 1) * (2 * index))
            summand += term

        return summand

    def rate(self, step):
        if step < self.warming_steps:
            return self.maximal_rate * (step + 1) / self.warming_steps

        progress = min(1, (step - self.warming_steps) / max(1, self.total_steps - self.warming_steps))

        return self.minimal_rate + (self.maximal_rate - self.minimal_rate) * 0.5 * (1 + self.cosine(3.141592653589793 * progress))
