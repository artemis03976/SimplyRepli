class EarlyStopping:
    def __init__(self, patience=3, delta=0, mode='max'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score

        # min for loss-like metrics, max for accuracy-like metrics
        elif (self.mode == 'min' and val_score < self.best_score - self.delta) or \
             (self.mode == 'max' and val_score > self.best_score + self.delta):
            # reset for new best score
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            # trigger early stopping
            if self.counter >= self.patience:
                self.early_stop = True
