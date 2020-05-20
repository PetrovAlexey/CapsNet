import torch

CONFIG_LOG_MESSAGE = 'Rotation={}; steps={}; initial_lr={}; model={}; \'' \
                     'n_workers={}; dataset={}'

MODELS = [
    'resnet18'
]

MODEL_PATH = '.\\best_models\\best_{}'

batch_size_train = 64
batch_size_test = 1000

random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

momentum = 0.9
log_interval = 10

init_lr = 0.01