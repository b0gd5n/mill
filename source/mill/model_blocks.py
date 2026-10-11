from .model_parameters import Parameter
from .model_pipeline import Model

class Dense:
    def __init__(self, blocks_distributor, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

        self.weights = Parameter(values=blocks_distributor.get_weights(inputs=inputs, outputs=outputs), regularized=True)
        self.biases = Parameter(values=blocks_distributor.get_biases(quantity=outputs), regularized=False)

        self.inputs_tensors = []

    def predict_tensor(self, inputs_tensor, mode='inference'):
        if mode == 'training':
            self.inputs_tensors.append(inputs_tensor)

        outputs_tensor = []

        for batch in range(len(inputs_tensor)):
            outputs_tensor.append([])

            for node in range(self.outputs):
                summand = self.biases.values[node]

                for value in range(self.inputs):
                    summand += inputs_tensor[batch][value] * self.weights.values[node * self.inputs + value]

                outputs_tensor[batch].append(summand)

        return outputs_tensor

    def calculate_gradients(self, gradients_tensor):
        inputs_tensor = self.inputs_tensors.pop()

        inputs_gradients_tensor = []

        for batch in range(len(gradients_tensor)):
            inputs_gradients_tensor.append([])

            for node in range(self.outputs):
                self.biases.gradients[node] += gradients_tensor[batch][node]

                for value in range(self.inputs):
                    self.weights.gradients[node * self.inputs + value] += gradients_tensor[batch][node] * inputs_tensor[batch][value]

            for value in range(self.inputs):
                summand = 0

                for node in range(self.outputs):
                    summand += gradients_tensor[batch][node] * self.weights.values[node * self.inputs + value]

                inputs_gradients_tensor[batch].append(summand)

        return inputs_gradients_tensor

    def get_parameters(self):
        return [self.weights, self.biases]

class LayerNormalization:
    def __init__(self, inputs, blocks_distributor, epsilon=1e-5):
        self.inputs = inputs

        self.scales = Parameter(values=blocks_distributor.get_scales(quantity=inputs), regularized=False)
        self.shifts = Parameter(values=blocks_distributor.get_shifts(quantity=inputs), regularized=False)

        self.normalized_tensors = []
        self.deviations_lists = []

        self.epsilon = epsilon

    def predict_tensor(self, inputs_tensor, mode='inference'):
        normalized_tensor = []
        deviations_list = []

        outputs_tensor = []

        for batch in range(len(inputs_tensor)):
            normalized_tensor.append([])
            outputs_tensor.append([])

            mean = 0

            for value in range(self.inputs):
                mean += inputs_tensor[batch][value]

            mean /= self.inputs

            variance = 0

            for value in range(self.inputs):
                variance += (inputs_tensor[batch][value] - mean) * (inputs_tensor[batch][value] - mean)

            deviations_list.append((variance / self.inputs + self.epsilon) ** 0.5)

            for value in range(self.inputs):
                normalized_tensor[batch].append((inputs_tensor[batch][value] - mean) / deviations_list[batch])
                outputs_tensor[batch].append(self.scales.values[value] * normalized_tensor[batch][value] + self.shifts.values[value])

        if mode == 'training':
            self.normalized_tensors.append(normalized_tensor)
            self.deviations_lists.append(deviations_list)

        return outputs_tensor

    def calculate_gradients(self, gradients_tensor):
        normalized_tensor = self.normalized_tensors.pop()
        deviations_list = self.deviations_lists.pop()

        inputs_gradients_tensor = []

        for batch in range(len(gradients_tensor)):
            inputs_gradients_tensor.append([])

            gradient_summand = 0
            weighted_summand = 0

            for value in range(self.inputs):
                self.scales.gradients[value] += gradients_tensor[batch][value] * normalized_tensor[batch][value]
                self.shifts.gradients[value] += gradients_tensor[batch][value]

                gradient_summand += gradients_tensor[batch][value] * self.scales.values[value]
                weighted_summand += gradients_tensor[batch][value] * self.scales.values[value] * normalized_tensor[batch][value]

            for value in range(self.inputs):
                inputs_gradients_tensor[batch].append((self.inputs * gradients_tensor[batch][value] * self.scales.values[value] - gradient_summand - normalized_tensor[batch][value] * weighted_summand) / (self.inputs * deviations_list[batch]))

        return inputs_gradients_tensor

    def get_parameters(self):
        return [self.scales, self.shifts]

class BatchNormalization:
    def __init__(self, inputs, blocks_distributor, momentum=1e-1, epsilon=1e-5):
        self.inputs = inputs

        self.scales = Parameter(values=blocks_distributor.get_scales(quantity=inputs), regularized=False)
        self.shifts = Parameter(values=blocks_distributor.get_shifts(quantity=inputs), regularized=False)

        self.running_means = Parameter(values=blocks_distributor.get_running_means(quantity=inputs), regularized=False)
        self.running_variances = Parameter(values=blocks_distributor.get_running_variances(quantity=inputs), regularized=False)

        self.normalized_tensors = []
        self.deviations_lists = []

        self.momentum = momentum
        self.epsilon = epsilon

    def predict_tensor(self, inputs_tensor, mode='inference'):
        outputs_tensor = []

        for batch in range(len(inputs_tensor)):
            outputs_tensor.append([])

            for value in range(self.inputs):
                outputs_tensor[batch].append(0)

        if mode == 'training':
            normalized_tensor = []
            deviations_list = []

            for batch in range(len(inputs_tensor)):
                normalized_tensor.append([])

                for value in range(self.inputs):
                    normalized_tensor[batch].append(0)

            for value in range(self.inputs):
                mean = 0

                for batch in range(len(inputs_tensor)):
                    mean += inputs_tensor[batch][value]

                mean /= len(inputs_tensor)

                variance = 0

                for batch in range(len(inputs_tensor)):
                    variance += (inputs_tensor[batch][value] - mean) * (inputs_tensor[batch][value] - mean)

                variance /= len(inputs_tensor)

                deviations_list.append((variance + self.epsilon) ** 0.5)

                self.running_means.values[value] = (1 - self.momentum) * self.running_means.values[value] + self.momentum * mean

                if len(inputs_tensor) > 1:
                    self.running_variances.values[value] = (1 - self.momentum) * self.running_variances.values[value] + self.momentum * variance * len(inputs_tensor) / (len(inputs_tensor) - 1)

                for batch in range(len(inputs_tensor)):
                    normalized_tensor[batch][value] = (inputs_tensor[batch][value] - mean) / deviations_list[value]
                    outputs_tensor[batch][value] = self.scales.values[value] * normalized_tensor[batch][value] + self.shifts.values[value]

            self.normalized_tensors.append(normalized_tensor)
            self.deviations_lists.append(deviations_list)
        else:
            for value in range(self.inputs):
                for batch in range(len(inputs_tensor)):
                    outputs_tensor[batch][value] = self.scales.values[value] * (inputs_tensor[batch][value] - self.running_means.values[value]) / (self.running_variances.values[value] + self.epsilon) ** 0.5 + self.shifts.values[value]

        return outputs_tensor

    def calculate_gradients(self, gradients_tensor):
        normalized_tensor = self.normalized_tensors.pop()
        deviations_list = self.deviations_lists.pop()

        inputs_gradients_tensor = []

        for batch in range(len(gradients_tensor)):
            inputs_gradients_tensor.append([])

            for value in range(self.inputs):
                inputs_gradients_tensor[batch].append(0)

        for value in range(self.inputs):
            gradient_summand = 0
            weighted_summand = 0

            for batch in range(len(gradients_tensor)):
                self.scales.gradients[value] += gradients_tensor[batch][value] * normalized_tensor[batch][value]
                self.shifts.gradients[value] += gradients_tensor[batch][value]

                gradient_summand += gradients_tensor[batch][value] * self.scales.values[value]
                weighted_summand += gradients_tensor[batch][value] * self.scales.values[value] * normalized_tensor[batch][value]

            for batch in range(len(gradients_tensor)):
                inputs_gradients_tensor[batch][value] = (len(gradients_tensor) * gradients_tensor[batch][value] * self.scales.values[value] - gradient_summand - normalized_tensor[batch][value] * weighted_summand) / (len(gradients_tensor) * deviations_list[value])

        return inputs_gradients_tensor

    def get_parameters(self):
        return [self.scales, self.shifts]

    def get_buffers(self):
        return [self.running_means, self.running_variances]

class InvertedDropout:
    def __init__(self, random_value_generator, probability=1e-1):
        self.random_value_generator = random_value_generator
        self.probability = probability

        self.masks_tensors = []

    def predict_tensor(self, inputs_tensor, mode='inference'):
        if mode != 'training' or self.probability == 0:
            return inputs_tensor

        mask_tensor = []
        outputs_tensor = []

        for batch in range(len(inputs_tensor)):
            mask_tensor.append([])
            outputs_tensor.append([])

            for value in range(len(inputs_tensor[batch])):
                if self.random_value_generator.get_uniform_value(minimal_value=0, maximal_value=1) < self.probability:
                    mask_tensor[batch].append(0)
                else:
                    mask_tensor[batch].append(1 / (1 - self.probability))

                outputs_tensor[batch].append(inputs_tensor[batch][value] * mask_tensor[batch][value])

        self.masks_tensors.append(mask_tensor)

        return outputs_tensor

    def calculate_gradients(self, gradients_tensor):
        if self.probability == 0:
            return gradients_tensor

        mask_tensor = self.masks_tensors.pop()

        inputs_gradients_tensor = []

        for batch in range(len(gradients_tensor)):
            inputs_gradients_tensor.append([])

            for value in range(len(gradients_tensor[batch])):
                inputs_gradients_tensor[batch].append(gradients_tensor[batch][value] * mask_tensor[batch][value])

        return inputs_gradients_tensor

class Residual:
    def __init__(self, *model_blocks):
        self.branch = Model(*model_blocks)

    def predict_tensor(self, inputs_tensor, mode='inference'):
        branch_outputs_tensor = self.branch.predict_tensor(inputs_tensor=inputs_tensor, mode=mode)

        outputs_tensor = []

        for batch in range(len(inputs_tensor)):
            outputs_tensor.append([])

            for value in range(len(inputs_tensor[batch])):
                outputs_tensor[batch].append(inputs_tensor[batch][value] + branch_outputs_tensor[batch][value])

        return outputs_tensor

    def calculate_gradients(self, gradients_tensor):
        branch_inputs_gradients_tensor = self.branch.calculate_gradients(gradients_tensor=gradients_tensor)

        inputs_gradients_tensor = []

        for batch in range(len(gradients_tensor)):
            inputs_gradients_tensor.append([])

            for value in range(len(gradients_tensor[batch])):
                inputs_gradients_tensor[batch].append(gradients_tensor[batch][value] + branch_inputs_gradients_tensor[batch][value])

        return inputs_gradients_tensor

    def get_parameters(self):
        return self.branch.get_parameters()

    def get_buffers(self):
        return self.branch.get_buffers()

class RectifiedLinearUnit:
    def __init__(self, alpha=0):
        self.alpha = alpha

        self.inputs_tensors = []

    def predict_tensor(self, inputs_tensor, mode='inference'):
        if mode == 'training':
            self.inputs_tensors.append(inputs_tensor)

        outputs_tensor = []

        for batch in range(len(inputs_tensor)):
            outputs_tensor.append([])

            for value in range(len(inputs_tensor[batch])):
                if inputs_tensor[batch][value] > 0:
                    outputs_tensor[batch].append(inputs_tensor[batch][value])
                else:
                    outputs_tensor[batch].append(self.alpha * inputs_tensor[batch][value])

        return outputs_tensor

    def calculate_gradients(self, gradients_tensor):
        inputs_tensor = self.inputs_tensors.pop()

        inputs_gradients_tensor = []

        for batch in range(len(gradients_tensor)):
            inputs_gradients_tensor.append([])

            for value in range(len(gradients_tensor[batch])):
                if inputs_tensor[batch][value] > 0:
                    inputs_gradients_tensor[batch].append(gradients_tensor[batch][value])
                else:
                    inputs_gradients_tensor[batch].append(self.alpha * gradients_tensor[batch][value])

        return inputs_gradients_tensor
