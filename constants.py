CHECKPOINT_DIR = "_fastflow_experiment_checkpoints"
SAVE_DIR = "./_results"

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
NUM_EPOCHS = 150
LR = 1e-3
BACKBONE_LR = 1e-4
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9

LOG_INTERVAL = 10
EVAL_INTERVAL = 1
CHECKPOINT_INTERVAL = 10

TRAINING_BACKBONE = True
