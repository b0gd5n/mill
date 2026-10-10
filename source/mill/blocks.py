from .parameters import Parameter

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

class RectifiedLinearUnit:
    def __init__(self):
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
                    output[batch].append(0)

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
                    output[batch].append(0)

        return output

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