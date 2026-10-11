class Model:
    def __init__(self, *model_blocks):
        self.model_blocks = model_blocks

    def predict_tensor(self, inputs_tensor, mode='inference'):
        current_tensor = inputs_tensor

        for block in range(len(self.model_blocks)):
            current_tensor = self.model_blocks[block].predict_tensor(inputs_tensor=current_tensor, mode=mode)

        return current_tensor

    def calculate_gradients(self, gradients_tensor):
        current_tensor = gradients_tensor

        for block in reversed(range(len(self.model_blocks))):
            current_tensor = self.model_blocks[block].calculate_gradients(gradients_tensor=current_tensor)

        return current_tensor

    def get_parameters(self):
        parameters = []

        for block in range(len(self.model_blocks)):
            if hasattr(self.model_blocks[block], 'get_parameters'):
                block_parameters = self.model_blocks[block].get_parameters()

                for parameter in range(len(block_parameters)):
                    duplicated = False

                    for existing_parameter in range(len(parameters)):
                        if block_parameters[parameter] is parameters[existing_parameter]:
                            duplicated = True

                    if not duplicated:
                        parameters.append(block_parameters[parameter])

        return parameters

    def get_buffers(self):
        buffers = []

        for block in range(len(self.model_blocks)):
            if hasattr(self.model_blocks[block], 'get_buffers'):
                block_buffers = self.model_blocks[block].get_buffers()

                for buffer in range(len(block_buffers)):
                    buffers.append(block_buffers[buffer])

        return buffers

    def save_model(self, path='model.txt'):
        parameters = self.get_parameters() + self.get_buffers()

        with open(path, 'w') as file:
            for parameter in range(len(parameters)):
                line = ''

                for value in range(len(parameters[parameter].values)):
                    if value > 0:
                        line += ' '

                    line += repr(float(parameters[parameter].values[value]))

                file.write(line + '\n')

    def load_model(self, path='model.txt'):
        with open(path, 'r') as file:
            lines = file.read().split('\n')

        parameters = self.get_parameters() + self.get_buffers()

        for parameter in range(len(parameters)):
            numbers = lines[parameter].split()

            for value in range(len(parameters[parameter].values)):
                parameters[parameter].values[value] = float(numbers[value])
