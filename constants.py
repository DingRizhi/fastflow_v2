CHECKPOINT_DIR = "_experiment_checkpoints"
SAVE_DIR = "_results"

MVTEC_CATEGORIES = [
    "bottle",
    "amz_1_down",
    "amz_1_right",
]

BACKBONE_DEIT = "deit_base_distilled_patch16_384"
BACKBONE_CAIT = "cait_m48_448"
BACKBONE_RESNET18 = "resnet18"
BACKBONE_WIDE_RESNET50 = "wide_resnet50_2"

SUPPORTED_BACKBONES = [
    BACKBONE_DEIT,
    BACKBONE_CAIT,
    BACKBONE_RESNET18,
    BACKBONE_WIDE_RESNET50,
]

BATCH_SIZE = 32
NUM_EPOCHS = 200
LR = 1e-3
BACKBONE_LR = 1e-3
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9

LOG_INTERVAL = 10
EVAL_INTERVAL = 1
CHECKPOINT_INTERVAL = 10

TRAINING_BACKBONE = False

LOAD_PRETRAINED_BACKBONE = False
PRETRAINED_BACKBONE_PTH = "/home/log/PycharmProjects/fastflow_v2/_experiment_checkpoints/exp11_resnet18_2022-12-01-11-43/last_model.pth"

SCHEDULER_EPOTCH = 5

HIGH_RESOLUTION_RESULT = True

