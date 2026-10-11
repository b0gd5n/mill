class Uniform:
    def __init__(self, random_value_generator, minimal_value=-0.5, maximal_value=0.5):
        self.random_value_generator = random_value_generator

        self.minimal_value = minimal_value
        self.maximal_value = maximal_value

    def get_weights(self, inputs, outputs):
        weights = []

        for weight in range(inputs * outputs):
            weights.append(self.random_value_generator.get_uniform_value(minimal_value=self.minimal_value, maximal_value=self.maximal_value))

        return weights

    def get_biases(self, quantity):
        biases = []

        for bias in range(quantity):
            biases.append(0)

        return biases

    def get_scales(self, quantity):
        scales = []

        for scale in range(quantity):
            scales.append(1)

        return scales

    def get_shifts(self, quantity):
        shifts = []

        for shift in range(quantity):
            shifts.append(0)

        return shifts

    def get_running_means(self, quantity):
        running_means = []

        for running_mean in range(quantity):
            running_means.append(0)

        return running_means

    def get_running_variances(self, quantity):
        running_variances = []

        for running_variance in range(quantity):
            running_variances.append(1)

        return running_variances

class Normal:
    def __init__(self, random_value_generator, mean=0, deviation=2e-2):
        self.random_value_generator = random_value_generator

        self.mean = mean
        self.deviation = deviation

    def get_weights(self, inputs, outputs):
        weights = []

        while len(weights) < inputs * outputs:
            first_uniform_value = self.random_value_generator.get_uniform_value(minimal_value=-1, maximal_value=1)
            second_uniform_value = self.random_value_generator.get_uniform_value(minimal_value=-1, maximal_value=1)

            radius = first_uniform_value * first_uniform_value + second_uniform_value * second_uniform_value

            if radius <= 0 or radius >= 1:
                continue

            mantissa = radius
            logarithm = 0

            while mantissa < 1:
                mantissa *= 2
                logarithm -= 0.6931

            power = (mantissa - 1) / (mantissa + 1)
            ratio_squared = power * power

            for index in range(30):
                logarithm += 2 * power / (2 * index + 1)
                power *= ratio_squared

            factor = (-2 * logarithm / radius) ** 0.5

            weights.append(self.mean + self.deviation * first_uniform_value * factor)

            if len(weights) < inputs * outputs:
                weights.append(self.mean + self.deviation * second_uniform_value * factor)

        return weights

    def get_biases(self, quantity):
        biases = []

        for bias in range(quantity):
            biases.append(0)

        return biases

    def get_scales(self, quantity):
        scales = []

        for scale in range(quantity):
            scales.append(1)

        return scales

    def get_shifts(self, quantity):
        shifts = []

        for shift in range(quantity):
            shifts.append(0)

        return shifts

    def get_running_means(self, quantity):
        running_means = []

        for running_mean in range(quantity):
            running_means.append(0)

        return running_means

    def get_running_variances(self, quantity):
        running_variances = []

        for running_variance in range(quantity):
            running_variances.append(1)

        return running_variances
