
class ValidationLossEarlyStopping:
    def __init__(self, model, evaluator, patience=8, minimum_improvement=0):
        self.model = model
        self.evaluator = evaluator
        self.patience = patience
        self.minimum_improvement = minimum_improvement

        self.best_loss = float('inf')
        self.best_epoch = -1
        self.best_parameters = []
        self.epochs_without_improvement = 0
        self.epoch = 0

    def check(self, inputs, targets):
        loss = self.evaluator.evaluate(predictions=self.model.forward(tensor=inputs), targets=targets)

        if loss < self.best_loss - self.minimum_improvement:
            self.best_loss = loss
            self.best_epoch = self.epoch
            self.epochs_without_improvement = 0

            self.remember()
        else:
            self.epochs_without_improvement += 1

        self.epoch += 1

        return self.epochs_without_improvement >= self.patience

    def remember(self):
        parameters = self.model.parameters()

        self.best_parameters = []

        for parameter in range(len(parameters)):
            self.best_parameters.append([])

            for value in range(len(parameters[parameter].values)):
                self.best_parameters[parameter].append(parameters[parameter].values[value])

    def restore(self):
        parameters = self.model.parameters()

        for parameter in range(len(parameters)):
            for value in range(len(parameters[parameter].values)):
                parameters[parameter].values[value] = self.best_parameters[parameter][value]
