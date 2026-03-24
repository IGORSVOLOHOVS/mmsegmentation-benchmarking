import optuna
from mmengine.hooks import Hook
from mmseg.registry import HOOKS

@HOOKS.register_module()
class OptunaPruningHook(Hook):
    def __init__(self, trial, metric='val/mDice', interval=1):
        self.trial = trial
        self.metric = metric
        self.interval = interval

    def after_val_epoch(self, runner, metrics=None):
        if metrics is not None and self.metric in metrics:
            current_score = metrics[self.metric]
        else:
            current_score = runner.message_hub.get_scalar(self.metric).current()

        step = runner.epoch if runner.train_loop.by_epoch else runner.iter
        self.trial.report(current_score, step)

        if self.trial.should_prune():
            message = f"Trial {self.trial.number} pruned at step {step}"
            raise optuna.TrialPruned(message)
