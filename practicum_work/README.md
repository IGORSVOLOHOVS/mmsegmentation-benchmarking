# Выбор и обучение модели из mmsegmentation для задачи мультиклассовой семантической сегментации
Целевая метрика — mDice score. Проект считается выполненным при mDIce > 0.75.

## Этап 1. Исследовательский анализ (EDA)

### Анализ качества данных 
Исходный датасет был проверен на наличие изображений без соответствующих меток с помощью скрипта `dataset.eda.missed_labels`. Ошибок разметки (отсутствующих масок) не обнаружено. 
Формат датасета был конвертирован в COCO для стандартизации работы с `mmsegmentation`.

### EDA
Анализ распределения классов (в пикселях):
- Background: 11908826 pixels
- Cat: 653268 pixels
- Dog: 545106 pixels
В датасете наблюдается сильный дисбаланс в сторону фона (background).

Средний размер изображений: 256x256. В датасете присутствуют только изображения этого размера (min=256x256, max=256x256).

[Jupyter Notebook с EDA](notebook.ipynb) приложен в репозитории.

## Этап 2. Формирование первичных гипотез

### Стартовая гипотеза 1: DeepLabV3+ с ResNet-50 (d8)

**Результаты обучения**  
- [Конфиг](../configs/experiments/deeplabv3plus_r50_d8_20k_coco_animals.py)
- [ClearML](https://app.clear.ml/projects/ef8fb423fbc040078cf0a46d46db7a45/experiments/a1b7ce551b7c4223844b2ca5a68742c0/output/log)

**Анализ качества**  
Метрики на тестовой выборке:

| Class | mIoU | mAcc | mDice |
|---|---|---|---|
| background | 97.37 | 98.96 | 98.67 |
| cat | 75.91 | 86.68 | 86.31 |
| dog | 69.52 | 76.24 | 82.02 |
| **All** | **80.94** | **87.29** | **89.00** |

Модель показывает неплохие метрики, но есть сложности с сегментацией собак (mDice = 82.02).

### Стартовая гипотеза 2: DeepLabV3+ с ResNet-50 (d16)

**Результаты обучения**  
- [Конфиг](../configs/experiments/deeplabv3plus_r50_d16_20k_coco_animals.py)
- [ClearML](https://app.clear.ml/projects/ef8fb423fbc040078cf0a46d46db7a45/experiments/0d4a3cf7c7da49e48ea14aea41c580f6/output/log)

**Анализ качества**  
Метрики на тестовой выборке:

| Class | mIoU | mAcc | mDice |
|---|---|---|---|
| background | 97.01 | 98.83 | 98.48 |
| cat | 74.97 | 83.24 | 85.70 |
| dog | 66.84 | 77.63 | 80.12 |
| **All** | **79.61** | **86.57** | **88.10** |

Увеличение dilation до `d16` на ResNet-50 незначительно ухудшило результат модели (mDice снизился до 88.10), объекты стали распознаваться чуть хуже.

## Этап 3. Эксперименты по улучшению качества

### Эксперимент 1: Усложнение архитектуры (ResNet-101)

**Результаты обучения**
- [Конфиг](../configs/experiments/deeplabv3plus_r101_d16_20k_coco_animals.py)
- [ClearML](https://app.clear.ml/projects/ef8fb423fbc040078cf0a46d46db7a45/experiments/7aef4d0b58ff45339bafa439d126197b/output/log) 

**Анализ качества**
Метрики на тестовой выборке:

| Class | mIoU | mAcc | mDice |
|---|---|---|---|
| background | 97.26 | 98.96 | 98.61 |
| cat | 76.28 | 86.19 | 86.55 |
| dog | 66.56 | 74.47 | 79.93 |
| **All** | **80.04** | **86.54** | **88.36** |

Усложнение архитектуры до ResNet-101 дало небольшое улучшение относительно ResNet-50 с dilation=16. Однако результат все еще хуже бейзлайна (ResNet-50, d=8) по общей метрике mDice (88.36 vs 89.00).

## Этап 4. Заключение и выбор лучшего эксперимента

### Лучший эксперимент 
Лучший эксперимент по метрике mDice: **Стартовая гипотеза 1 (DeepLabV3+, ResNet-50, d=8)**.
Оказалось, что для данной задачи и выбранного размера картинок (256x256), сложный бэкбон (ResNet-101) и большое рецептивное поле (d16) не были необходимы. В итоге сработал самый базовый и легкий конфиг, который лучше всего обобщается на тестовой выборке.

**mDice (test subset) = 89.00**  
[ClearML](https://app.clear.ml/projects/ef8fb423fbc040078cf0a46d46db7a45/experiments/a1b7ce551b7c4223844b2ca5a68742c0/output/log)

### Примеры корректных предсказаний (тестовый датасет)
Отобранные изображения с лучшим индивидуальным mDice score (отсортированы скриптом):  
![Best 1](experiments/deeplabv3plus_r50_d8/best_worst_test/best/best_0_score_0.9858_000000437537_2563.png)  
![Best 2](experiments/deeplabv3plus_r50_d8/best_worst_test/best/best_1_score_0.9857_000000495159_4697.png)  
![Best 3](experiments/deeplabv3plus_r50_d8/best_worst_test/best/best_2_score_0.9853_000000446604_4215.png)  
![Best 4](experiments/deeplabv3plus_r50_d8/best_worst_test/best/best_3_score_0.9852_000000414495_3471.png)  

### Примеры ошибок (тестовый датасет)
Отобранные изображения с худшим индивидуальным mDice score:  
![Worst 1](experiments/deeplabv3plus_r50_d8/best_worst_test/worst/worst_0_score_0.0000_000000284884_6459.png)  
![Worst 2](experiments/deeplabv3plus_r50_d8/best_worst_test/worst/worst_1_score_0.0000_000000436539_4321.png)  
![Worst 3](experiments/deeplabv3plus_r50_d8/best_worst_test/worst/worst_2_score_0.1320_000000445187_3686.png)  
![Worst 4](experiments/deeplabv3plus_r50_d8/best_worst_test/worst/worst_3_score_0.2722_000000308083_5809.png)  

## Этап 5. Документация кода

```text
.
├── configs
│   ├── _base_
│   │   ├── datasets
│   │   │   └── coco_animals.py - базовый конфиг датасета COCOAnimalsDataset
│   │   └── models
│   │       ├── deeplabv3plus_r50-d8.py - базовый конфиг модели ResNet-50 d8
│   │       ├── deeplabv3plus_r50-d16.py - базовый конфиг модели ResNet-50 d16
│   │       └── deeplabv3plus_r101-d16.py - базовый конфиг модели ResNet-101 d16
│   └── experiments - директория с запускаемыми конфигами для всех экспериментов
├── mmseg
│   └── datasets
│       ├── __init__.py - регистрация датасета COCOAnimalsDataset
│       └── coco_animals.py - класс COCOAnimalsDataset
└── practicum_work
    ├── src
    │   ├── dataset
    │   │   ├── coco.py - генерация датасета в формате COCO
    │   │   ├── eda.py - генерация метрик EDA, распределения классов и размеров
    │   │   └── label.py - наложение цветных масок на изображения датасета
    │   └── analysis 
    │       ├── dump_model_predictions.py - скрипт для пакетного инференса и сохранения предсказаний модели
    │       └── save_best_on_worst_based_on_individual_dice_score.py - скрипт для расчета mDice и отбора лучших/худших примеров для визуализации
```

# Repo
[https://github.com/IGORSVOLOHOVS/mmsegmentation-benchmarking](https://github.com/IGORSVOLOHOVS/mmsegmentation-benchmarking)

<!-- ### Возможности для улучшения 
Решил я поэкспериментировать с разными конфигурациями моделей с использованием хука на оптуну и ИИ. Запустил 15 экспериментов, 5 архитектур. Запустил, вот что получилось: -->