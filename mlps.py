import torch.nn as nn
import torch
from torch.func import functional_call
import numpy as np
from collections import deque
from mupify import mupify, rescale


import copy
import torch
import torch.nn as nn

class UncenteredMLP(nn.Module):
    def __init__(self, d_in=1, width=4096, depth=2, d_out=1, bias=True):
        super().__init__()
        self.input_layer = nn.Linear(d_in, width, bias)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(width, width, bias) for _ in range(depth - 1)]
        )
        self.output_layer = nn.Linear(width, d_out, bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            h = self.relu(layer(h))
        return self.output_layer(h)
    
    def get_activations(self, x):
        h_acts = []
        h = self.relu(self.input_layer(x))
        h_acts.append(h)
        for layer in self.hidden_layers:
            h = self.relu(layer(h))
            h_acts.append(h)
        h_out = self.output_layer(h)
        return h_acts, h_out

class MLP(nn.Module):
    """
    y(x) = model(x) - baseline(x), where `baseline` is a frozen clone of `model` at init.
    Optional: keep baseline in lower precision via `baseline_dtype` (e.g., torch.bfloat16).
    """
    def __init__(self, d_in=1, width=4096, depth=2, d_out=1, bias=True, baseline_dtype=None):
        super().__init__()
        self.model = UncenteredMLP(d_in, width, depth, d_out, bias)
        self.baseline = copy.deepcopy(self.model)  # snapshot at init
        for p in self.baseline.parameters():
            p.requires_grad = False
        self.baseline.eval()

        self._baseline_dtype = baseline_dtype
        if baseline_dtype is not None:
            self.baseline.to(dtype=baseline_dtype)

    @torch.no_grad()
    def recenter(self):
        """Reset the baseline to the current model weights (still frozen)."""
        self.baseline.load_state_dict(self.model.state_dict())
        if self._baseline_dtype is not None:
            self.baseline.to(dtype=self._baseline_dtype)
        self.baseline.eval()

    def forward(self, x):
        y = self.model(x)  # grads retained
        with torch.inference_mode():  # no grads/memory for baseline pass
            y0 = self.baseline(x)
        return y - y0

    def get_activations(self, x):
        h_acts = []
        h = self.relu(self.input_layer(x))
        h_acts.append(h)
        for layer in self.hidden_layers:
            h = self.relu(layer(h))
            h_acts.append(h)
        h_out = self.output_layer(h)
        return h_acts, h_out


def train_network(model, batch_function, lr=1e-2, max_iter=int(1e3), loss_checkpoints=None, percent_thresholds=None,
                  gamma=1., ema_smoother=0.0, X_tr=None, y_tr=None, X_te=None, y_te=None, stopper=None, only_thresholds=False,
                  **kwargs):
    """
    Returns:
        model, losses, timekeys

    - timekeys[j]: first gradient step where the loss drops below the j-th threshold.
    - If RELATIVE mode, thresholds are `percent_thresholds * init_loss`.
    - If ABSOLUTE mode, thresholds are raw absolutes and the comparison metric is raw loss.
    - If only_thresholds is True, only returns the timekeys and not the full loss curves.
    """
    #checking if losses are absolute or relative
    has_abs = (loss_checkpoints is not None) and len(loss_checkpoints) > 0
    is_relative = (percent_thresholds is not None) and len(percent_thresholds) > 0
    if has_abs and is_relative:
        raise ValueError("Provide exactly one of loss_checkpoints OR percent_thresholds.")
    if not has_abs and not is_relative:
        raise ValueError("You must provide one of loss_checkpoints or percent_thresholds.")

    # model stuff
    lr = lr * gamma if gamma >= 1 else lr * (gamma**2.)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    mupify(model, opt, param="mup")
    rescale(model, gamma)
    loss_fn = torch.nn.MSELoss()

    # thresholding
    thresholds = np.asarray(percent_thresholds if is_relative else loss_checkpoints, dtype=float)
    thresholds = np.sort(thresholds)[::-1] # descending
    timekeys = np.full(thresholds.shape, 0, dtype=int)

    if not(only_thresholds):
        tr_losses = np.empty(max_iter, dtype=float)
        te_losses = np.empty(max_iter, dtype=float)
    ema = None
    pointer = 0   

    # training loop 
    for i in range(max_iter):
        X_tr, y_tr = batch_function(i)
    
        opt.zero_grad()
        out = model(X_tr)
        loss = loss_fn(out, y_tr)
        loss.backward()
        opt.step()

        tr_loss_val = float(loss.item())
        if X_tr is None: #online training if none
            te_loss_val = tr_loss_val
        else:
            with torch.no_grad():
                # X_te, y_te = map(ensure_torch, (X_te, y_te))
                out = model(X_te)
                loss = loss_fn(out, y_te)
                te_loss_val = float(loss.item())

        # initialize thresholds & loss trace baseline at first step
        if i == 0:
            ema_tr = tr_loss_val
            ema_te = te_loss_val
            if is_relative:
                thresholds *= tr_loss_val
            # prefill losses after init val calculated
            if not(only_thresholds):
                tr_losses[:] = tr_loss_val
                te_losses[:] = te_loss_val

        ema_tr = (ema_smoother * ema_tr + (1.0 - ema_smoother) * tr_loss_val)
        ema_te = (ema_smoother * ema_te + (1.0 - ema_smoother) * te_loss_val)
        if not(only_thresholds):
            tr_losses[i] = ema_tr
            te_losses[i] = ema_te

        if stopper is not None:
            stop, _ = stopper.update(i, te_loss_val)
            if stop and not only_thresholds:
                tr_losses[i:] = tr_losses[i]
                te_losses[i:] = te_losses[i]
                return {"model": model, "train_losses": tr_losses, "test_losses": te_losses, "timekeys": timekeys}
            elif stop:
                return {"model": model, "train_losses": ema_tr, "test_losses": ema_te, "timekeys": timekeys}
        while pointer < len(thresholds) and ema_tr < thresholds[pointer]:
            timekeys[pointer] = i
            if not(only_thresholds):
                tr_losses[i:] = tr_losses[i]
                te_losses[i:] = te_losses[i]
            pointer += 1

        # early exit if all thresholds crossed
        if pointer == len(thresholds):
            if only_thresholds:
                return {"model": model, "train_losses": ema_tr, "test_losses": ema_te, "timekeys": timekeys}
            return {"model": model, "train_losses": tr_losses, "test_losses": te_losses, "timekeys": timekeys}

    if only_thresholds:
        return {"model": model, "train_losses": ema_tr, "test_losses": ema_te, "timekeys": timekeys}
    return {"model": model, "train_losses": tr_losses, "test_losses": te_losses, "timekeys": timekeys}


# for checking when training has plateaued; discard if we want the full training run
# as this takes longer to evaluate than just running to completion 
class PlateauStopper:
    def __init__(self, min_iter=200, min_drop_frac=0.1, window=50, eval_every=10, patience=2, slope_eps=5e-4, rel_improve_eps=5e-4,
                 ema_smoother=0.0, use_log_time=True):
        self.min_iter = int(min_iter)
        self.min_drop_frac = float(min_drop_frac)
        self.window = int(window)
        self.eval_every = int(eval_every)
        self.patience = int(patience)
        self.slope_eps = float(slope_eps)
        self.rel_improve_eps = float(rel_improve_eps)
        self.ema_smoother = float(ema_smoother)
        self.use_log_time = bool(use_log_time)

        self.init_loss = None
        self.best_loss = np.inf
        self.started = False
        self.history = deque(maxlen=self.window)   # (step, loss) pairs
        self.loss = None
        self._last_check_step = -np.inf
        self._flat_hits = 0

    def update(self, step: int, test_loss: float):
        if self.loss is None:
            self.loss = float(test_loss)
        self.loss = self.ema_smoother * self.loss + (1 - self.ema_smoother) * float(test_loss)

        if self.init_loss is None:
            self.init_loss = self.loss
        self.best_loss = min(self.best_loss, self.loss)

        # don't start tracking until some time has passed/some improvement in losss
        if step < self.min_iter:
            self.history.append((step, self.loss))
            return False, {"reason": "warmup", "step": step}
        if not self.started:
            self.started = (self.best_loss <= (1 - self.min_drop_frac) * self.init_loss)

        self.history.append((step, self.loss))

        # only check every eval_every steps with a full window
        if step - self._last_check_step < self.eval_every or len(self.history) < self.window:
            return False, {"reason": "not_due_or_short_window", "step": step}

        self._last_check_step = step

        # build x=log(step) or x=step, y=loss
        steps = np.array([s for s, _ in self.history], dtype=float)
        ys    = np.array([y for _, y in self.history], dtype=float)

        x = np.log(steps + 1.0) if self.use_log_time else steps
        # simple linear regression slope on window
        x_mean = x.mean()
        y_mean = ys.mean()
        denom = np.sum((x - x_mean) ** 2)
        slope = 0.0 if denom == 0 else np.sum((x - x_mean) * (ys - y_mean)) / denom

        # relative improvement across the window
        y0, yT = ys[0], ys[-1]
        rel_improve = max(0.0, (y0 - yT) / max(abs(y0), 1e-8))

        plateau = self.started and (abs(slope) < self.slope_eps) and (rel_improve < self.rel_improve_eps)

        if plateau:
            self._flat_hits += 1
        else:
            self._flat_hits = 0

        should_stop = self._flat_hits >= self.patience

        info = {
            "slope": slope,
            "rel_improve": rel_improve,
            "flat_hits": self._flat_hits,
            "started": self.started,
            "loss": self.loss,
            "step": step
        }
        return should_stop, info

    def reset(self):
        self.init_loss = None
        self.best_loss = np.inf
        self.started = False
        self.history.clear()
        self.ema_loss = None
        self._last_check_step = -np.inf
        self._flat_hits = 0
        return self