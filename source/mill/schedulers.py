class Cosine:
    def __init__(self, minimal_rate, maximal_rate, warming_steps, total_steps, iterations=16):
        self.minimal_rate = minimal_rate
        self.maximal_rate = maximal_rate
        self.warming_steps = warming_steps
        self.total_steps = total_steps
        self.iterations = iterations

    def cosine(self, angle):
        summation = 1
        term = 1

        for index in range(1, self.iterations):
            term *= -angle * angle / ((2 * index - 1) * (2 * index))
            summation += term

        return summation

    def rate(self, step):
        if step < self.warming_steps:
            return self.minimal_rate + (self.maximal_rate - self.minimal_rate) * step / self.warming_steps

        return self.minimal_rate + (self.maximal_rate - self.minimal_rate) * 0.5 * (1 + self.cosine(3.1415 * min(1, (step - self.warming_steps) / max(1, self.total_steps - self.warming_steps))))
