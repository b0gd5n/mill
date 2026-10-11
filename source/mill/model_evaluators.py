class MeanSquaredError:
    def __init__(self, model):
        self.model = model

    def get_loss(self, predictions_tensor, targets_tensor):
        summand = 0

        for batch in range(len(predictions_tensor)):
            for value in range(len(predictions_tensor[batch])):
                summand += (predictions_tensor[batch][value] - targets_tensor[batch][value]) ** 2

        return summand / (len(predictions_tensor) * len(predictions_tensor[0]))

    def get_validation_loss(self, inputs_tensor, targets_tensor):
        return self.get_loss(predictions_tensor=self.model.predict_tensor(inputs_tensor=inputs_tensor, mode='inference'), targets_tensor=targets_tensor)

    def get_loss_gradients(self, predictions_tensor, targets_tensor):
        loss_gradients_tensor = []

        for batch in range(len(predictions_tensor)):
            loss_gradients_tensor.append([])

            for value in range(len(predictions_tensor[batch])):
                loss_gradients_tensor[batch].append(2 * (predictions_tensor[batch][value] - targets_tensor[batch][value]) / (len(predictions_tensor) * len(predictions_tensor[0])))

        return loss_gradients_tensor
