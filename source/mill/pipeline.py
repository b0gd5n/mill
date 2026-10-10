
class Sequential:
    def __init__(self, *blocks):
        self.blocks = blocks

    def forward(self, tensor, mode='inference'):
        current = tensor

        for block in range(len(self.blocks)):
            current = self.blocks[block].forward(current, mode=mode)

        return current

    def backward(self, tensor):
        current = tensor

        for block in reversed(range(len(self.blocks))):
            current = self.blocks[block].backward(current)

        return current

    def parameters(self):
        parameters = []

        for block in range(len(self.blocks)):
            if hasattr(self.blocks[block], 'parameters'):
                block_parameters = self.blocks[block].parameters()

                for parameter in range(len(block_parameters)):
                    if not any(block_parameters[parameter] is parameters[existing] for existing in range(len(parameters))):
                        parameters.append(block_parameters[parameter])

        return parameters

    def save(self, path='model.txt'):
        parameters = self.parameters()

        with open(path, 'w') as file:
            for parameter in range(len(parameters)):
                line = ''

                for value in range(len(parameters[parameter].values)):
                    if value > 0:
                        line += ' '

                    line += repr(float(parameters[parameter].values[value]))

                file.write(line + '\n')

    def load(self, path='model.txt'):
        with open(path, 'r') as file:
            lines = file.read().split('\n')

        parameters = self.parameters()

        for parameter in range(len(parameters)):
            numbers = lines[parameter].split()

            for value in range(len(parameters[parameter].values)):
                parameters[parameter].values[value] = float(numbers[value])
