import argparse
from main import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow")
    parser.add_argument(
        "-cfg", "--config", default='configs/resnet18.yaml', type=str,  help="path to config file",
    )
    parser.add_argument("--data", type=str, default='/data/BYD_dingzi/dataset', help="path to mvtec folder",)
    parser.add_argument(
        "-cat",
        "--category",
        default='duanziqiliui_crop_v2',
        type=str,
        help="category name",
    )
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint",
        default=""
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    print(f"-------------eval-------------")
    evaluate(args)
