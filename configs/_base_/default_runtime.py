"""Shared runtime settings from the original paper experiments."""

weight = None
resume = False
evaluate = True
test_only = False
seed = None
num_worker = 8
batch_size = 8
gradient_accumulation_steps = 1
batch_size_val = None
batch_size_test = None
clip_grad = None
sync_bn = False
enable_amp = True
amp_dtype = 'float16'
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
enable_wandb = False
enable_swanlab = False
wandb_project = 'COVAL'
wandb_key = None
mix_prob = 0
param_dicts = None

hooks = [
    dict(type='CheckpointLoader'),
    dict(type='ModelHook'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=False),
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)
