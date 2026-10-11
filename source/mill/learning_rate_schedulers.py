class Cosine:
    def __init__(self, minimal_learning_rate, maximal_learning_rate, warming_steps, total_steps, series_terms=16):
        self.minimal_learning_rate = minimal_learning_rate
        self.maximal_learning_rate = maximal_learning_rate

        self.warming_steps = warming_steps
        self.total_steps = total_steps

        self.series_terms = series_terms

    def get_cosine(self, angle):
        summand = 1
        term = 1

        for index in range(1, self.series_terms):
            term *= -angle * angle / ((2 * index - 1) * (2 * index))

            summand += term

        return summand

    def get_learning_rate(self, step):
        if step < self.warming_steps:
            return self.minimal_learning_rate + (self.maximal_learning_rate - self.minimal_learning_rate) * step / self.warming_steps

        return self.minimal_learning_rate + (self.maximal_learning_rate - self.minimal_learning_rate) * 0.5 * (1 + self.get_cosine(3.1415 * min(1, (step - self.warming_steps) / max(1, self.total_steps - self.warming_steps))))
