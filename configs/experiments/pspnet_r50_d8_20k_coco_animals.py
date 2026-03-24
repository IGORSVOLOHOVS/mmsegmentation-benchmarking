_base_ = [
    '../_base_/models/pspnet_r50-d8.py',
    '../_base_/datasets/coco_animals.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

data_root = 'practicum_work/dataset'
dataset_type = 'COCOAnimalsDataset'
crop_size = (256, 256)

data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=3,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=1.0)
        ]),
    auxiliary_head=dict(
        num_classes=3,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=0.4)
        ]))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(img_path='base/img/train', seg_map_path='base/labels/train'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(img_path='base/img/val', seg_map_path='base/labels/val'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'], prefix='val')
test_evaluator = val_evaluator

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='ClearMLVisBackend', init_kwargs=dict(project_name='mmseg-benchmarking', task_name='pspnet_r50_d8'))
]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
