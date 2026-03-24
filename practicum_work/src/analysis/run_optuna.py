import optuna
import pandas as pd
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmseg.registry import HOOKS
import os

# ==========================================
# 1. ОПРЕДЕЛЯЕМ ХУК (ОБЯЗАТЕЛЬНО БЕЗ РЕГИСТРАЦИИ В CONFIG)
# ==========================================
class OptunaPruningHook(Hook):
    def __init__(self, trial, metric='val/mDice'):
        self.trial = trial
        self.metric = metric

    def after_val_epoch(self, runner, metrics=None):
        # Забираем метрику для Optuna
        if metrics is not None and self.metric in metrics:
            current_score = metrics[self.metric]
        else:
            # Пытаемся забрать из hub, если metrics пуст
            try:
                current_score = runner.message_hub.get_scalar(self.metric).current()
            except:
                return # Если метрики нет, ничего не делаем

        step = runner.epoch if runner.train_loop.by_epoch else runner.iter
        self.trial.report(current_score, step)

        # Если эксперимент идет хуже других — обрубаем его
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Trial {self.trial.number} pruned at step {step}")

# ==========================================
# 2. ФУНКЦИЯ ПОДБОРА ПАРАМЕТРОВ
# ==========================================
def objective(trial):
    # --- Выбор архитектуры ---
    arch_config = trial.suggest_categorical('config', [
        'configs/experiments/deeplabv3plus_r50_d8_20k_coco_animals.py',
        'configs/experiments/pspnet_r50_d8_20k_coco_animals.py',
        'configs/experiments/fcn_r50_d8_20k_coco_animals.py',
        'configs/experiments/upernet_r50_20k_coco_animals.py',
        'configs/experiments/segformer_mit-b0_20k_coco_animals.py'
    ])
    
    # --- Гиперпараметры ---
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    prob_flip = trial.suggest_float('prob_flip', 0.0, 1.0)
    
    # Загружаем конфиг
    cfg = Config.fromfile(arch_config)
    cfg.optim_wrapper.optimizer.lr = lr
    cfg.optim_wrapper.optimizer.weight_decay = weight_decay
    
    # Меняем вероятность RandomFlip
    if hasattr(cfg, 'train_pipeline'):
        for transform in cfg.train_pipeline:
            if transform.get('type') == 'RandomFlip':
                transform['prob'] = prob_flip
    
    # Указываем директорию триала
    cfg.work_dir = f'./work_dirs/optuna_study/trial_{trial.number}'
    
    # Быстрые настройки (уменьшаем количество итераций для подбора)
    cfg.train_cfg.max_iters = 6000
    cfg.train_cfg.val_interval = 2000
    cfg.default_hooks.checkpoint.interval = 6000
    
    # ВАЖНО: Мы не добавляем хук в cfg.custom_hooks, 
    # чтобы избежать ошибки форматирования "Failed to format the config file".
    # Вместо этого мы зарегистрируем инстанс хука напрямую в Runner.

    try:
        runner = Runner.from_cfg(cfg)
        
        # Регистрируем хук ВРУЧНУЮ после создания runner
        optuna_hook = OptunaPruningHook(trial=trial, metric='val/mDice')
        runner.register_hook(optuna_hook, priority='LOWEST')
        
        runner.train()
        
        # Получаем финальные метрики
        val_metrics = runner.val_loop.run()
        return val_metrics['mDice']
        
    except optuna.TrialPruned as e:
        # Модель была "отсечена" как бесперспективная
        raise e
    except Exception as e:
        print(f"Ошибка в триале {trial.number}: {e}")
        return 0.0

if __name__ == "__main__":
    study = optuna.create_study(
        study_name='mmseg_benchmark_tuning',
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2000)
    )
    
    print("Запуск Optuna Study для 5 архитектур...")
    study.optimize(objective, n_trials=15)
    
    print("\nЛучшие параметры:", study.best_params)
    
    df = study.trials_dataframe()
    df.to_csv('practicum_work/optuna_results.csv', index=False)
    print("Результаты сохранены в practicum_work/optuna_results.csv")
