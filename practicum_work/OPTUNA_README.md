# Инструкция по интеграции Optuna с MMSegmentation

Для эффективного подбора гиперпараметров (learning rate, weight decay, аугментации и др.) можно использовать библиотеку **Optuna**. Чтобы не тратить драгоценные GPU-часы на заведомо проигрышные эксперименты, мы настроим механизм **Pruning (отсечение)** — Optuna будет прерывать обучение модели, если её промежуточные метрики оказываются хуже, чем у предыдущих успешных запусков.

Для работы с MMSegmentation потребуется выполнить 3 основных шага.

## Шаг 1. Создание кастомного хука (OptunaPruningHook)

MMSegmentation работает через систему хуков (Hooks). Нам нужно создать класс, который будет забирать метрику после каждой эпохи валидации и передавать её в Optuna.

В любой из ваших скриптов (например, `practicum_work/src/analysis/optuna_hook.py`) добавьте следующий код:

```python
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
        # Получаем значение метрики
        if metrics is not None and self.metric in metrics:
            current_score = metrics[self.metric]
        else:
            current_score = runner.message_hub.get_scalar(self.metric).current()

        # Сообщаем результат в Optuna на текущем шаге
        step = runner.epoch if runner.train_loop.by_epoch else runner.iter
        self.trial.report(current_score, step)

        # Проверка на прунинг (сохраняем ли этот эксперимент?)
        if self.trial.should_prune():
            message = f"Trial {self.trial.number} pruned at step {step}"
            raise optuna.TrialPruned(message)
```

## Шаг 2. Описание функции `objective`

Далее нужно написать функцию для самого подбора. Здесь мы загружаем наш базовый конфиг, подменяем в нём гиперпараметры значениями, которые предлагает Optuna, и запускаем `Runner`.

Создайте основной файл для запуска `practicum_work/src/analysis/run_optuna.py`:

```python
import optuna
from mmengine.config import Config
from mmengine.runner import Runner

# Важно: импортируйте хук, чтобы он зарегистрировался в системе HOOKS
from optuna_hook import OptunaPruningHook

def objective(trial):
    # 1. Предложения от Optuna 
    # (подбор learning rate по логарифмической шкале)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    # 2. Загружаем конфиг, который хотим улучшать
    cfg = Config.fromfile('configs/experiments/deeplabv3plus_r50_d8_20k_coco_animals.py')
    
    # 3. Модифицируем параметры внутри конфига
    cfg.optim_wrapper.optimizer.lr = lr
    cfg.optim_wrapper.optimizer.weight_decay = weight_decay
    
    # Указываем отдельную директорию для каждого триала
    cfg.work_dir = f'./work_dirs/optuna_study/trial_{trial.number}'
    
    # 4. Добавляем OptunaPruningHook
    optuna_hook = dict(
        type='OptunaPruningHook', 
        trial=trial, 
        metric='val/mDice' # Максимизируем mDice
    )
    
    if 'custom_hooks' not in cfg:
        cfg.custom_hooks = [optuna_hook]
    else:
        cfg.custom_hooks.append(optuna_hook)
        
    try:
        runner = Runner.from_cfg(cfg)
        runner.train()
        
        # Запускаем тест в конце и возвращаем метрику
        metrics = runner.test()
        return metrics['mDice']
        
    except optuna.TrialPruned as e:
        # Если Optuna решила отсечь триал, пробрасываем исключение
        raise e
    except Exception as e:
        print(f"Trial failed due to: {e}")
        return 0.0
```

## Шаг 3. Запуск Study (Обучения)

Инициализируем объект `study` и запускаем перебор. Добавьте в конец `run_optuna.py` следующий блок:

```python
if __name__ == "__main__":
    # Используем MedianPruner: если результаты триала ниже медианы
    # существующих результатов к этому шагу — прерываем его.
    # n_warmup_steps = 4000 защищает первые 4k итераций от раннего прунинга.
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=4000)
    
    study = optuna.create_study(
        study_name='mmseg_deeplab_tuning',
        direction='maximize', 
        pruner=pruner
    )
    
    # Запускаем 20 экспериментов
    study.optimize(objective, n_trials=20)

    print(f"Лучший результат: {study.best_value}")
    print(f"Лучшие гиперпараметры: {study.best_params}")
```

### Запуск скрипта
Запустите скрипт из корня проекта:
```bash
python practicum_work/src/analysis/run_optuna.py
```

### Что ещё можно тюнить:
```python
# Использовать или нет заморозку BatchNorm слоев:
backbone_norm_grad = trial.suggest_categorical('norm_grad', [True, False])
cfg.model.backbone.norm_cfg.requires_grad = backbone_norm_grad

# Менять ли аугментации:
prob_flip = trial.suggest_float('prob_flip', 0.0, 0.5)
# ... и подменить это значение в `cfg.train_pipeline`
```
