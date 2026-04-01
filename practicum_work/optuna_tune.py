import optuna
import pandas as pd
import os
import sys

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmseg.registry import HOOKS

# ==========================================
# 1. ОПРЕДЕЛЯЕМ ХУК БЕЗ РЕГИСТРАЦИИ В CONFIG
# ==========================================
class OptunaPruningHook(Hook):
    def __init__(self, trial, metric='val/mDice', interval=1):
        self.trial = trial
        self.metric = metric

    def after_val_epoch(self, runner, metrics=None):
        # Забираем метрику для Optuna
        if metrics is not None and self.metric in metrics:
            current_score = metrics[self.metric]
        else:
            try:
                current_score = runner.message_hub.get_scalar(self.metric).current()
            except:
                return # Если метрики еще нет, пропускаем

        step = runner.epoch if runner.train_loop.by_epoch else runner.iter
        self.trial.report(current_score, step)

        # Если эксперимент идет хуже других — обрубаем его
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Trial {self.trial.number} pruned at step {step}")


# ==========================================
# 2. ФУНКЦИЯ ПОДБОРА ПАРАМЕТРОВ
# ==========================================
def objective(trial):
    # --- Выбор архитектуры из 5 доступных ---
    arch_config = trial.suggest_categorical('config', [
        'configs/experiments/deeplabv3plus_r50_d8_20k_coco_animals.py',
        'configs/experiments/pspnet_r50_d8_20k_coco_animals.py',
        'configs/experiments/fcn_r50_d8_20k_coco_animals.py',
        'configs/experiments/upernet_r50_20k_coco_animals.py',
        'configs/experiments/segformer_mit-b0_20k_coco_animals.py'
    ])
    
    # --- Подбор гиперпараметров ---
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    prob_flip = trial.suggest_float('prob_flip', 0.0, 1.0)
    
    # Загружаем конфиг
    # Если запуск из корня, путь верный. Если из practicum_work, надо добавить ../
    if not os.path.exists(arch_config) and os.path.exists(os.path.join('..', arch_config)):
        arch_config = os.path.join('..', arch_config)
    
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
    
    # Настройки для ускорения перебора
    cfg.train_cfg.max_iters = 6000
    cfg.train_cfg.val_interval = 2000
    cfg.default_hooks.checkpoint.interval = 6000
    
    # === ОБУЧАЕМ И ПОЛУЧАЕМ ОЦЕНКУ ===
    try:
        runner = Runner.from_cfg(cfg)
        
        # РЕГИСТРИРУЕМ ХУК ВНУТРЬ РАННЕРА
        optuna_hook = OptunaPruningHook(trial=trial, metric='val/mDice')
        runner.register_hook(optuna_hook, priority='LOWEST')
        
        runner.train()
        
        # Получаем финальный mDice
        val_metrics = runner.val_loop.run()
        return val_metrics['mDice']
        
    except optuna.TrialPruned as e:
        # Успешно обрубили плохой эксперимент
        raise e
    except Exception as e:
        print(f"Эксперимент №{trial.number} сломался с ошибкой: {e}")
        return 0.0


# ==========================================
# 3. ЗАПУСК ИССЛЕДОВАНИЯ
# ==========================================
def main():
    study = optuna.create_study(
        study_name='mmseg_benchmark_tuner',
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2000)
    )

    print("Начинаю перебор гиперпараметров для 5 архитектур (15 попыток)...")
    study.optimize(objective, n_trials=15)

    # ==========================================
    # 4. ВЫВОД РЕЗУЛЬТАТОВ
    # ==========================================
    print("\n🏆 ЛУЧШИЕ ПАРАМЕТРЫ:")
    print(study.best_params)

    df = study.trials_dataframe()
    cols = ['number', 'value', 'state', 'params_config', 'params_lr', 'params_prob_flip', 'params_weight_decay']
    df_filtered = df[cols].sort_values(by='value', ascending=False)
    
    print("\nTop Trials:")
    print(df_filtered.to_string())
    
    # Сохраняем результат в CSV
    os.makedirs('practicum_work', exist_ok=True)
    df.to_csv('practicum_work/optuna_results.csv', index=False)
    print("\nResults saved to practicum_work/optuna_results.csv")

if __name__ == "__main__":
    main()
