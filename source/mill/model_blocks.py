from .model_parameters import Parameter
from .model_pipeline import Model

class Dense:
    def __init__(self, inputs, outputs, blocks_distributor):
        self.inputs = inputs
        self.outputs = outputs

        self.weights = Parameter(values=blocks_distributor.get_weights(inputs=inputs, outputs=outputs), regularized=True)
        self.biases = Parameter(values=blocks_distributor.get_biases(quantity=outputs), regularized=False)

        self.inputs_tensors = []

    def predict_tensor(self, inputs_tensor, mode='inference'):
        if mode == 'training':
            self.inputs_tensors.append(inputs_tensor)

        outputs_tensor = []

        self.transform_tensor(inputs_tensor=inputs_tensor, outputs_tensor=outputs_tensor)

        return outputs_tensor

    def calculate_gradients(self, gradients_tensor):
        inputs_gradients_tensor = []

        self.calculate_transformation_gradients(gradients_tensor=gradients_tensor, inputs_tensor=self.inputs_tensors.pop(), inputs_gradients_tensor=inputs_gradients_tensor)

        return inputs_gradients_tensor

    def transform_tensor(self, inputs_tensor, outputs_tensor):
        for element in range(len(inputs_tensor)):
            outputs_tensor.append([])

            if isinstance(inputs_tensor[element][0], list):
                self.transform_tensor(inputs_tensor=inputs_tensor[element], outputs_tensor=outputs_tensor[element])
            else:
                for node in range(self.outputs):
                    summand = self.biases.values[node]

                    for value in range(self.inputs):
                        summand += inputs_tensor[element][value] * self.weights.values[node * self.inputs + value]

                    outputs_tensor[element].append(summand)

    def calculate_transformation_gradients(self, gradients_tensor, inputs_tensor, inputs_gradients_tensor):
        for element in range(len(gradients_tensor)):
            inputs_gradients_tensor.append([])

            if isinstance(gradients_tensor[element][0], list):
                self.calculate_transformation_gradients(gradients_tensor=gradients_tensor[element], inputs_tensor=inputs_tensor[element], inputs_gradients_tensor=inputs_gradients_tensor[element])
            else:
                for node in range(self.outputs):
                    self.biases.gradients[node] += gradients_tensor[element][node]

                    for value in range(self.inputs):
                        self.weights.gradients[node * self.inputs + value] += gradients_tensor[element][node] * inputs_tensor[element][value]

                for value in range(self.inputs):
                    summand = 0

                    for node in range(self.outputs):
                        summand += gradients_tensor[element][node] * self.weights.values[node * self.inputs + value]

                    inputs_gradients_tensor[element].append(summand)

    def get_parameters(self):
        return [self.weights, self.biases]

class BatchNormalisation:
    def __init__(self, inputs, blocks_distributor, momentum=1e-1, epsilon=1e-5):
        self.inputs = inputs

        self.scales = Parameter(values=blocks_distributor.get_scales(quantity=inputs), regularized=False)
        self.shifts = Parameter(values=blocks_distributor.get_shifts(quantity=inputs), regularized=False)

        self.running_means = Parameter(values=blocks_distributor.get_running_means(quantity=inputs), regularized=False)
        self.running_variances = Parameter(values=blocks_distributor.get_running_variances(quantity=inputs), regularized=False)

        self.normalized_channels_lists = []
        self.deviations_lists = []

        self.momentum = momentum
        self.epsilon = epsilon

    def predict_tensor(self, inputs_tensor, mode='inference'):
        inputs_channels = self.collect_channels(tensor=inputs_tensor)
        outputs_channels = []

        if mode == 'training':
            normalized_channels = []
            deviations_list = []

            for channel in range(self.inputs):
                normalized_channels.append([])
                outputs_channels.append([])

                mean = 0

                for position in range(len(inputs_channels[channel])):
                    mean += inputs_channels[channel][position]

                mean /= len(inputs_channels[channel])

                variance = 0

                for position in range(len(inputs_channels[channel])):
                    variance += (inputs_channels[channel][position] - mean) * (inputs_channels[channel][position] - mean)

                variance /= len(inputs_channels[channel])

                deviations_list.append((variance + self.epsilon) ** 0.5)

                self.running_means.values[channel] = (1 - self.momentum) * self.running_means.values[channel] + self.momentum * mean

                if len(inputs_channels[channel]) > 1:
                    self.running_variances.values[channel] = (1 - self.momentum) * self.running_variances.values[channel] + self.momentum * variance * len(inputs_channels[channel]) / (len(inputs_channels[channel]) - 1)

                for position in range(len(inputs_channels[channel])):
                    normalized_channels[channel].append((inputs_channels[channel][position] - mean) / deviations_list[channel])
                    outputs_channels[channel].append(self.scales.values[channel] * normalized_channels[channel][position] + self.shifts.values[channel])

            self.normalized_channels_lists.append(normalized_channels)
            self.deviations_lists.append(deviations_list)
        else:
            for channel in range(self.inputs):
                outputs_channels.append([])

                for position in range(len(inputs_channels[channel])):
                    outputs_channels[channel].append(self.scales.values[channel] * (inputs_channels[channel][position] - self.running_means.values[channel]) / (self.running_variances.values[channel] + self.epsilon) ** 0.5 + self.shifts.values[channel])

        return self.restore_tensor(template_tensor=inputs_tensor, channels=outputs_channels)

    def calculate_gradients(self, gradients_tensor):
        normalized_channels = self.normalized_channels_lists.pop()
        deviations_list = self.deviations_lists.pop()

        gradients_channels = self.collect_channels(tensor=gradients_tensor)
        inputs_gradients_channels = []

        for channel in range(self.inputs):
            inputs_gradients_channels.append([])

            gradient_summand = 0
            weighted_summand = 0

            for position in range(len(gradients_channels[channel])):
                self.scales.gradients[channel] += gradients_channels[channel][position] * normalized_channels[channel][position]
                self.shifts.gradients[channel] += gradients_channels[channel][position]

                gradient_summand += gradients_channels[channel][position] * self.scales.values[channel]
                weighted_summand += gradients_channels[channel][position] * self.scales.values[channel] * normalized_channels[channel][position]

            for position in range(len(gradients_channels[channel])):
                inputs_gradients_channels[channel].append((len(gradients_channels[channel]) * gradients_channels[channel][position] * self.scales.values[channel] - gradient_summand - normalized_channels[channel][position] * weighted_summand) / (len(gradients_channels[channel]) * deviations_list[channel]))

        return self.restore_tensor(template_tensor=gradients_tensor, channels=inputs_gradients_channels)

    def collect_channels(self, tensor):
        channels = []

        for channel in range(self.inputs):
            channels.append([])

        for batch in range(len(tensor)):
            for channel in range(self.inputs):
                self.collect_values(tensor=tensor[batch][channel], values=channels[channel])

        return channels

    def collect_values(self, tensor, values):
        if isinstance(tensor, list):
            for value in range(len(tensor)):
                self.collect_values(tensor=tensor[value], values=values)
        else:
            values.append(tensor)

    def restore_tensor(self, template_tensor, channels):
        positions = []

        for channel in range(self.inputs):
            positions.append(0)

        restored_tensor = []

        for batch in range(len(template_tensor)):
            restored_tensor.append([])

            for channel in range(self.inputs):
                restored_tensor[batch].append(self.restore_values(template_tensor=template_tensor[batch][channel], channels=channels, positions=positions, channel=channel))

        return restored_tensor

    def restore_values(self, template_tensor, channels, positions, channel):
        if isinstance(template_tensor, list):
            restored_tensor = []

            for value in range(len(template_tensor)):
                restored_tensor.append(self.restore_values(template_tensor=template_tensor[value], channels=channels, positions=positions, channel=channel))

            return restored_tensor

        positions[channel] += 1

        return channels[channel][positions[channel] - 1]

    def get_parameters(self):
        return [self.scales, self.shifts]

    def get_buffers(self):
        return [self.running_means, self.running_variances]

class LayerNormalisation:
    def __init__(self, inputs, blocks_distributor, epsilon=1e-5):
        self.inputs = inputs

        self.scales = Parameter(values=blocks_distributor.get_scales(quantity=inputs), regularized=False)
        self.shifts = Parameter(values=blocks_distributor.get_shifts(quantity=inputs), regularized=False)

        self.normalized_tensors = []
        self.deviations_tensors = []

        self.epsilon = epsilon

    def predict_tensor(self, inputs_tensor, mode='inference'):
        outputs_tensor = []
        normalized_tensor = []
        deviations_tensor = []

        self.normalize_tensor(inputs_tensor=inputs_tensor, outputs_tensor=outputs_tensor, normalized_tensor=normalized_tensor, deviations_tensor=deviations_tensor)

        if mode == 'training':
            self.normalized_tensors.append(normalized_tensor)
            self.deviations_tensors.append(deviations_tensor)

        return outputs_tensor

    def calculate_gradients(self, gradients_tensor):
        inputs_gradients_tensor = []

        self.calculate_normalization_gradients(gradients_tensor=gradients_tensor, normalized_tensor=self.normalized_tensors.pop(), deviations_tensor=self.deviations_tensors.pop(), inputs_gradients_tensor=inputs_gradients_tensor)

        return inputs_gradients_tensor

    def normalize_tensor(self, inputs_tensor, outputs_tensor, normalized_tensor, deviations_tensor):
        for element in range(len(inputs_tensor)):
            outputs_tensor.append([])
            normalized_tensor.append([])

            if isinstance(inputs_tensor[element][0], list):
                deviations_tensor.append([])

                self.normalize_tensor(inputs_tensor=inputs_tensor[element], outputs_tensor=outputs_tensor[element], normalized_tensor=normalized_tensor[element], deviations_tensor=deviations_tensor[element])
            else:
                mean = 0

                for value in range(self.inputs):
                    mean += inputs_tensor[element][value]

                mean /= self.inputs

                variance = 0

                for value in range(self.inputs):
                    variance += (inputs_tensor[element][value] - mean) * (inputs_tensor[element][value] - mean)

                deviations_tensor.append((variance / self.inputs + self.epsilon) ** 0.5)

                for value in range(self.inputs):
                    normalized_tensor[element].append((inputs_tensor[element][value] - mean) / deviations_tensor[element])
                    outputs_tensor[element].append(self.scales.values[value] * normalized_tensor[element][value] + self.shifts.values[value])

    def calculate_normalization_gradients(self, gradients_tensor, normalized_tensor, deviations_tensor, inputs_gradients_tensor):
        for element in range(len(gradients_tensor)):
            inputs_gradients_tensor.append([])

            if isinstance(gradients_tensor[element][0], list):
                self.calculate_normalization_gradients(gradients_tensor=gradients_tensor[element], normalized_tensor=normalized_tensor[element], deviations_tensor=deviations_tensor[element], inputs_gradients_tensor=inputs_gradients_tensor[element])
            else:
                gradient_summand = 0
                weighted_summand = 0

                for value in range(self.inputs):
                    self.scales.gradients[value] += gradients_tensor[element][value] * normalized_tensor[element][value]
                    self.shifts.gradients[value] += gradients_tensor[element][value]

                    gradient_summand += gradients_tensor[element][value] * self.scales.values[value]
                    weighted_summand += gradients_tensor[element][value] * self.scales.values[value] * normalized_tensor[element][value]

                for value in range(self.inputs):
                    inputs_gradients_tensor[element].append((self.inputs * gradients_tensor[element][value] * self.scales.values[value] - gradient_summand - normalized_tensor[element][value] * weighted_summand) / (self.inputs * deviations_tensor[element]))

    def get_parameters(self):
        return [self.scales, self.shifts]

class InvertedDropout:
    def __init__(self, random_value_generator, probability=1e-1):
        self.random_value_generator = random_value_generator
        self.probability = probability

        self.masks_tensors = []

    def predict_tensor(self, inputs_tensor, mode='inference'):
        if mode != 'training' or self.probability == 0:
            return inputs_tensor

        self.masks_tensors.append(self.create_mask_tensor(inputs_tensor=inputs_tensor))

        return self.apply_mask_tensor(inputs_tensor=inputs_tensor, mask_tensor=self.masks_tensors[-1])

    def calculate_gradients(self, gradients_tensor):
        if self.probability == 0:
            return gradients_tensor

        return self.apply_mask_tensor(inputs_tensor=gradients_tensor, mask_tensor=self.masks_tensors.pop())

    def create_mask_tensor(self, inputs_tensor):
        mask_tensor = []

        for value in range(len(inputs_tensor)):
            if isinstance(inputs_tensor[value], list):
                mask_tensor.append(self.create_mask_tensor(inputs_tensor=inputs_tensor[value]))
            elif self.random_value_generator.get_uniform_value(minimal_value=0, maximal_value=1) < self.probability:
                mask_tensor.append(0)
            else:
                mask_tensor.append(1 / (1 - self.probability))

        return mask_tensor

    def apply_mask_tensor(self, inputs_tensor, mask_tensor):
        outputs_tensor = []

        for value in range(len(inputs_tensor)):
            if isinstance(inputs_tensor[value], list):
                outputs_tensor.append(self.apply_mask_tensor(inputs_tensor=inputs_tensor[value], mask_tensor=mask_tensor[value]))
            else:
                outputs_tensor.append(inputs_tensor[value] * mask_tensor[value])

        return outputs_tensor

class Residual:
    def __init__(self, *model_blocks):
        self.branch = Model(*model_blocks)

    def predict_tensor(self, inputs_tensor, mode='inference'):
        return self.add_tensors(first_tensor=inputs_tensor, second_tensor=self.branch.predict_tensor(inputs_tensor=inputs_tensor, mode=mode))

    def calculate_gradients(self, gradients_tensor):
        return self.add_tensors(first_tensor=gradients_tensor, second_tensor=self.branch.calculate_gradients(gradients_tensor=gradients_tensor))

    def add_tensors(self, first_tensor, second_tensor):
        outputs_tensor = []

        for value in range(len(first_tensor)):
            if isinstance(first_tensor[value], list):
                outputs_tensor.append(self.add_tensors(first_tensor=first_tensor[value], second_tensor=second_tensor[value]))
            else:
                outputs_tensor.append(first_tensor[value] + second_tensor[value])

        return outputs_tensor

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

        return self.activate_tensor(inputs_tensor=inputs_tensor)

    def calculate_gradients(self, gradients_tensor):
        return self.calculate_activation_gradients(inputs_tensor=self.inputs_tensors.pop(), gradients_tensor=gradients_tensor)

    def activate_tensor(self, inputs_tensor):
        outputs_tensor = []

        for value in range(len(inputs_tensor)):
            if isinstance(inputs_tensor[value], list):
                outputs_tensor.append(self.activate_tensor(inputs_tensor=inputs_tensor[value]))
            elif inputs_tensor[value] > 0:
                outputs_tensor.append(inputs_tensor[value])
            else:
                outputs_tensor.append(self.alpha * inputs_tensor[value])

        return outputs_tensor

    def calculate_activation_gradients(self, inputs_tensor, gradients_tensor):
        inputs_gradients_tensor = []

        for value in range(len(inputs_tensor)):
            if isinstance(inputs_tensor[value], list):
                inputs_gradients_tensor.append(self.calculate_activation_gradients(inputs_tensor=inputs_tensor[value], gradients_tensor=gradients_tensor[value]))
            elif inputs_tensor[value] > 0:
                inputs_gradients_tensor.append(gradients_tensor[value])
            else:
                inputs_gradients_tensor.append(self.alpha * gradients_tensor[value])

        return inputs_gradients_tensor
