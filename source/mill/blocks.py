from .parameters import Parameter
from .pipeline import Sequential

class Dense:
    def __init__(self, inputs, outputs, distributor):
        self.inputs = inputs
        self.outputs = outputs

        self.weights = Parameter(values=distributor.generate(inputs=inputs, outputs=outputs))
        self.biases = Parameter(values=[], regularized=False)
        self.last_inputs = []

        for bias in range(outputs):
            self.biases.values.append(0)
            self.biases.gradients.append(0)

    def forward(self, tensor, mode='inference'):
        if mode == 'training':
            self.last_inputs.append(tensor)

        output = []

        for batch in range(len(tensor)):
            output.append([])

            for node in range(self.outputs):
                summand = self.biases.values[node]

                for value in range(self.inputs):
                    summand += tensor[batch][value] * self.weights.values[node * self.inputs + value]

                output[batch].append(summand)

        return output

    def backward(self, tensor):
        last_input = self.last_inputs.pop()

        output = []

        for batch in range(len(tensor)):
            output.append([])

            for node in range(self.outputs):
                self.biases.gradients[node] += tensor[batch][node]

                for value in range(self.inputs):
                    self.weights.gradients[node * self.inputs + value] += tensor[batch][node] * last_input[batch][value]

            for value in range(self.inputs):
                summand = 0

                for node in range(self.outputs):
                    summand += tensor[batch][node] * self.weights.values[node * self.inputs + value]

                output[batch].append(summand)

        return output

    def parameters(self):
        return [self.weights, self.biases]

class LayerNormalization:
    def __init__(self, features, epsilon=1e-5):
        self.features = features
        self.epsilon = epsilon

        self.scales = Parameter(values=[], regularized=False)
        self.shifts = Parameter(values=[], regularized=False)

        self.last_normalized = []
        self.last_deviations = []

        for feature in range(features):
            self.scales.values.append(1)
            self.scales.gradients.append(0)
            self.shifts.values.append(0)
            self.shifts.gradients.append(0)

    def forward(self, tensor, mode='inference'):
        normalized = []
        deviations = []
        output = []

        for batch in range(len(tensor)):
            normalized.append([])
            output.append([])

            mean = 0

            for value in range(self.features):
                mean += tensor[batch][value]

            mean /= self.features

            variance = 0

            for value in range(self.features):
                variance += (tensor[batch][value] - mean) * (tensor[batch][value] - mean)

            deviations.append((variance / self.features + self.epsilon) ** 0.5)

            for value in range(self.features):
                normalized[batch].append((tensor[batch][value] - mean) / deviations[batch])
                output[batch].append(self.scales.values[value] * normalized[batch][value] + self.shifts.values[value])

        if mode == 'training':
            self.last_normalized.append(normalized)
            self.last_deviations.append(deviations)

        return output

    def backward(self, tensor):
        last_normalized = self.last_normalized.pop()
        last_deviations = self.last_deviations.pop()

        output = []

        for batch in range(len(tensor)):
            output.append([])

            gradient_sum = 0
            weighted_sum = 0

            for value in range(self.features):
                self.scales.gradients[value] += tensor[batch][value] * last_normalized[batch][value]
                self.shifts.gradients[value] += tensor[batch][value]

                gradient_sum += tensor[batch][value] * self.scales.values[value]
                weighted_sum += tensor[batch][value] * self.scales.values[value] * last_normalized[batch][value]

            for value in range(self.features):
                output[batch].append((self.features * tensor[batch][value] * self.scales.values[value] - gradient_sum - last_normalized[batch][value] * weighted_sum) / (self.features * last_deviations[batch]))

        return output

    def parameters(self):
        return [self.scales, self.shifts]

class BatchNormalization:
    def __init__(self, features, momentum=0.1, epsilon=1e-5):
        self.features = features
        self.momentum = momentum
        self.epsilon = epsilon

        self.scales = Parameter(values=[], regularized=False)
        self.shifts = Parameter(values=[], regularized=False)

        self.running_means = Parameter(values=[], regularized=False)
        self.running_variances = Parameter(values=[], regularized=False)

        self.last_normalized = []
        self.last_deviations = []

        for feature in range(features):
            self.scales.values.append(1)
            self.scales.gradients.append(0)
            self.shifts.values.append(0)
            self.shifts.gradients.append(0)

            self.running_means.values.append(0)
            self.running_means.gradients.append(0)
            self.running_variances.values.append(1)
            self.running_variances.gradients.append(0)

    def forward(self, tensor, mode='inference'):
        output = []

        for batch in range(len(tensor)):
            output.append([])

            for value in range(self.features):
                output[batch].append(0)

        if mode == 'training':
            normalized = []
            deviations = []

            for batch in range(len(tensor)):
                normalized.append([])

                for value in range(self.features):
                    normalized[batch].append(0)

            for value in range(self.features):
                mean = 0

                for batch in range(len(tensor)):
                    mean += tensor[batch][value]

                mean /= len(tensor)

                variance = 0

                for batch in range(len(tensor)):
                    variance += (tensor[batch][value] - mean) * (tensor[batch][value] - mean)

                variance /= len(tensor)

                deviations.append((variance + self.epsilon) ** 0.5)

                self.running_means.values[value] = (1 - self.momentum) * self.running_means.values[value] + self.momentum * mean

                if len(tensor) > 1:
                    self.running_variances.values[value] = (1 - self.momentum) * self.running_variances.values[value] + self.momentum * variance * len(tensor) / (len(tensor) - 1)

                for batch in range(len(tensor)):
                    normalized[batch][value] = (tensor[batch][value] - mean) / deviations[value]
                    output[batch][value] = self.scales.values[value] * normalized[batch][value] + self.shifts.values[value]

            self.last_normalized.append(normalized)
            self.last_deviations.append(deviations)
        else:
            for value in range(self.features):
                for batch in range(len(tensor)):
                    output[batch][value] = self.scales.values[value] * (tensor[batch][value] - self.running_means.values[value]) / (self.running_variances.values[value] + self.epsilon) ** 0.5 + self.shifts.values[value]

        return output

    def backward(self, tensor):
        last_normalized = self.last_normalized.pop()
        last_deviations = self.last_deviations.pop()

        output = []

        for batch in range(len(tensor)):
            output.append([])

            for value in range(self.features):
                output[batch].append(0)

        for value in range(self.features):
            gradient_sum = 0
            weighted_sum = 0

            for batch in range(len(tensor)):
                self.scales.gradients[value] += tensor[batch][value] * last_normalized[batch][value]
                self.shifts.gradients[value] += tensor[batch][value]

                gradient_sum += tensor[batch][value] * self.scales.values[value]
                weighted_sum += tensor[batch][value] * self.scales.values[value] * last_normalized[batch][value]

            for batch in range(len(tensor)):
                output[batch][value] = (len(tensor) * tensor[batch][value] * self.scales.values[value] - gradient_sum - last_normalized[batch][value] * weighted_sum) / (len(tensor) * last_deviations[value])

        return output

    def parameters(self):
        return [self.scales, self.shifts]

    def buffers(self):
        return [self.running_means, self.running_variances]

class InvertedDropout:
    def __init__(self, generator, probability=0.5):
        self.generator = generator
        self.probability = probability

        self.last_masks = []

    def forward(self, tensor, mode='inference'):
        if mode != 'training' or self.probability == 0:
            return tensor

        mask = []
        output = []

        for batch in range(len(tensor)):
            mask.append([])
            output.append([])

            for value in range(len(tensor[batch])):
                if self.generator.generate(minimal_value=0, maximal_value=1) < self.probability:
                    mask[batch].append(0)
                else:
                    mask[batch].append(1 / (1 - self.probability))

                output[batch].append(tensor[batch][value] * mask[batch][value])

        self.last_masks.append(mask)

        return output

    def backward(self, tensor):
        if self.probability == 0:
            return tensor

        last_mask = self.last_masks.pop()

        output = []

        for batch in range(len(tensor)):
            output.append([])

            for value in range(len(tensor[batch])):
                output[batch].append(tensor[batch][value] * last_mask[batch][value])

        return output

class Residual:
    def __init__(self, *blocks):
        self.branch = Sequential(*blocks)

    def forward(self, tensor, mode='inference'):
        branch_output = self.branch.forward(tensor, mode=mode)

        output = []

        for batch in range(len(tensor)):
            output.append([])

            for value in range(len(tensor[batch])):
                output[batch].append(tensor[batch][value] + branch_output[batch][value])

        return output

    def backward(self, tensor):
        branch_gradient = self.branch.backward(tensor)

        output = []

        for batch in range(len(tensor)):
            output.append([])

            for value in range(len(tensor[batch])):
                output[batch].append(tensor[batch][value] + branch_gradient[batch][value])

        return output

    def parameters(self):
        return self.branch.parameters()

    def buffers(self):
        return self.branch.buffers()

class RectifiedLinearUnit:
    def __init__(self, alpha=0):
        self.alpha = alpha

        self.last_inputs = []

    def forward(self, tensor, mode='inference'):
        if mode == 'training':
            self.last_inputs.append(tensor)

        output = []

        for batch in range(len(tensor)):
            output.append([])

            for value in range(len(tensor[batch])):
                if tensor[batch][value] > 0:
                    output[batch].append(tensor[batch][value])
                else:
                    output[batch].append(self.alpha * tensor[batch][value])

        return output

    def backward(self, tensor):
        last_input = self.last_inputs.pop()

        output = []

        for batch in range(len(tensor)):
            output.append([])

            for value in range(len(tensor[batch])):
                if last_input[batch][value] > 0:
                    output[batch].append(tensor[batch][value])
                else:
                    output[batch].append(self.alpha * tensor[batch][value])

        return output
