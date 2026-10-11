class MeanSquaredError:
    def __init__(self, model):
        self.model = model

    def get_loss(self, predictions_tensor, targets_tensor):
        return self.summmarise_squared_errors(predictions_tensor=predictions_tensor, targets_tensor=targets_tensor) / self.count_values(tensor=predictions_tensor)

    def get_validation_loss(self, inputs_tensor, targets_tensor):
        return self.get_loss(predictions_tensor=self.model.predict_tensor(inputs_tensor=inputs_tensor, mode='inference'), targets_tensor=targets_tensor)

    def get_loss_gradients(self, predictions_tensor, targets_tensor):
        return self.calculate_squared_errors_gradients(predictions_tensor=predictions_tensor, targets_tensor=targets_tensor, quantity=self.count_values(tensor=predictions_tensor))

    def summmarise_squared_errors(self, predictions_tensor, targets_tensor):
        summand = 0

        for value in range(len(predictions_tensor)):
            if isinstance(predictions_tensor[value], list):
                summand += self.sum_squared_errors(predictions_tensor=predictions_tensor[value], targets_tensor=targets_tensor[value])
            else:
                summand += (predictions_tensor[value] - targets_tensor[value]) ** 2

        return summand

    def calculate_squared_errors_gradients(self, predictions_tensor, targets_tensor, quantity):
        loss_gradients_tensor = []

        for value in range(len(predictions_tensor)):
            if isinstance(predictions_tensor[value], list):
                loss_gradients_tensor.append(self.calculate_squared_errors_gradients(predictions_tensor=predictions_tensor[value], targets_tensor=targets_tensor[value], quantity=quantity))
            else:
                loss_gradients_tensor.append(2 * (predictions_tensor[value] - targets_tensor[value]) / quantity)

        return loss_gradients_tensor

    def count_values(self, tensor):
        quantity = 0

        for value in range(len(tensor)):
            if isinstance(tensor[value], list):
                quantity += self.count_values(tensor=tensor[value])
            else:
                quantity += 1

        return quantity
