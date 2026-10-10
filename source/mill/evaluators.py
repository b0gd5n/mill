class MeanSquaredError:
    def evaluate(self, predictions, targets):
        summand = 0

        for batch in range(len(predictions)):
            for value in range(len(predictions[batch])):
                summand += (predictions[batch][value] - targets[batch][value]) ** 2

        return summand / (len(predictions) * len(predictions[0]))

    def backward(self, predictions, targets):
        output = []

        for batch in range(len(predictions)):
            output.append([])

            for value in range(len(predictions[batch])):
                output[batch].append(2 * (predictions[batch][value] - targets[batch][value]) / (len(predictions) * len(predictions[0])))

        return output
