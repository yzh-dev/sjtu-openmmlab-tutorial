model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=1.0),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(type='LinearClsHead',
              num_classes=30, #输出分类数
              in_channels=1280,
              loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
              topk=(1, 5)),
)
# 预训练模型参数
load_from = 'mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(type='CustomDataset', # 不是在ImageNet上训练，需要将数据类型修改为CustomDataset
               data_prefix='data/fruit30_split/train',
               pipeline=[
                   dict(type='LoadImageFromFile'),
                   dict(type='RandomResizedCrop', size=224, backend='pillow'),
                   dict(type='RandomFlip',
                        flip_prob=0.5,
                        direction='horizontal'),
                   dict(type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                   dict(type='ImageToTensor', keys=['img']),
                   dict(type='ToTensor', keys=['gt_label']),
                   dict(type='Collect', keys=['img', 'gt_label'])
               ]),
    val=dict(type='CustomDataset',
             data_prefix='data/fruit30_split/val',
             pipeline=[
                 dict(type='LoadImageFromFile'),
                 dict(type='Resize', size=(256, -1), backend='pillow'),
                 dict(type='CenterCrop', crop_size=224),
                 dict(type='Normalize',
                      mean=[123.675, 116.28, 103.53],
                      std=[58.395, 57.12, 57.375],
                      to_rgb=True),
                 dict(type='ImageToTensor', keys=['img']),
                 dict(type='Collect', keys=['img'])
             ]),
    test=dict(type='CustomDataset',
              data_prefix='data/fruit30_split/val',
              pipeline=[
                  dict(type='LoadImageFromFile'),
                  dict(type='Resize', size=(256, -1), backend='pillow'),
                  dict(type='CenterCrop', crop_size=224),
                  dict(type='Normalize',
                       mean=[123.675, 116.28, 103.53],
                       std=[58.395, 57.12, 57.375],
                       to_rgb=True),
                  dict(type='ImageToTensor', keys=['img']),
                  dict(type='Collect', keys=['img'])
              ]),
)

# 配置优化器
evaluation = dict(interval=1, metric='accuracy')
# 原始训练是8卡训练，学习率为0.045，现在是1卡训练，因此将学习率除以8，约等于0.005
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=4e-05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', gamma=0.5, step=1)
# 原始300个epoch，修改为5
runner = dict(type='EpochBasedRunner', max_epochs=5)
checkpoint_config = dict(interval=5)
# interval从100修改到10，每10个epoch打印一次日志
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'

resume_from = None
workflow = [('train', 1)]
