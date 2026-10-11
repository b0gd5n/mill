class ValidationLossEarlyStopping:
    def __init__(self, model_evaluator, patience=8, minimal_improvement=0):
        self.patience = patience
        self.minimal_improvement = minimal_improvement

        self.model_evaluator = model_evaluator

        self.best_loss = float('inf')
        self.best_check = -1
        self.best_values = []

        self.checks = 0
        self.checks_without_improvement = 0

    def check_model(self, inputs_tensor, targets_tensor):
        loss = self.model_evaluator.get_validation_loss(inputs_tensor=inputs_tensor, targets_tensor=targets_tensor)

        if loss < self.best_loss - self.minimal_improvement:
            self.best_loss = loss
            self.best_check = self.checks
            self.checks_without_improvement = 0

            self.remember_model()
        else:
            self.checks_without_improvement += 1

        self.checks += 1

        return self.checks_without_improvement >= self.patience

    def remember_model(self):
        parameters = self.model_evaluator.model.get_parameters() + self.model_evaluator.model.get_buffers()

        self.best_values = []

        for parameter in range(len(parameters)):
            self.best_values.append([])

            for value in range(len(parameters[parameter].values)):
                self.best_values[parameter].append(parameters[parameter].values[value])

    def restore_model(self):
        parameters = self.model_evaluator.model.get_parameters() + self.model_evaluator.model.get_buffers()

        for parameter in range(len(parameters)):
            for value in range(len(parameters[parameter].values)):
                parameters[parameter].values[value] = self.best_values[parameter][value]
