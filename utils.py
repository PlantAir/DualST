import torch
import numpy as np
import random

def fix_random_seeds(seed):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class EarlyStopping(object):
    def __init__(self, model, save_path, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)
        self.model = model
        self.save_path = save_path

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics

            #save model
            torch.save(self.model.state_dict(), self.save_path)
            print('best model saved!')
        else:
            self.num_bad_epochs += 1
            print('Bad epochs nums:', self.num_bad_epochs)

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

def mean_absolute_percentage_error(y_true, y_pred):
    idx = np.nonzero(y_true)
    return np.mean(np.abs((y_true[idx] - y_pred[idx]) / y_true[idx]))

def compute_errors(preds, y_true):
    pred_mean = preds[:, 0:2]
    diff = y_true - pred_mean

    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))

    mape = mean_absolute_percentage_error(y_true.flatten(), pred_mean.flatten())

    return rmse, mae, mape


def valid(model, val_generator,criterion, device):
    model.eval()
    mape_loss = []
    mae_loss = []
    rmse_loss = []
    for i, (X_c, X_p, X_t, X_meta, Y_batch) in enumerate(val_generator):
        # Move tensors to the configured device
        X_c = X_c.type(torch.FloatTensor).to(device)
        X_p = X_p.type(torch.FloatTensor).to(device)
        X_t = X_t.type(torch.FloatTensor).to(device)
        X_meta = X_meta.type(torch.FloatTensor).to(device)

        # Forward pass
        outputs = model(X_c, X_p, X_t, X_meta)
        rmse, mae, mape = criterion(outputs.cpu().data.numpy(), Y_batch.data.numpy())

        mape_loss.append(mape)
        mae_loss.append(mae)
        rmse_loss.append(rmse)

    mape_loss = np.mean(mape_loss)
    mae_loss = np.mean(mae_loss)
    rmse_loss = np.mean(rmse_loss)

    return rmse_loss, mae_loss, mape_loss


def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a, b) / (a_norm * b_norm)
    return cos

if __name__ == '__main__':
    pass