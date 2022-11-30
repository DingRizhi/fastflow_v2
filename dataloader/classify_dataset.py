from torch.utils import data
import os
from PIL import Image
import glob


class ClassifyDataset(data.Dataset):

    def __init__(self, root, mode='train', transform=None):
        self.mode = mode
        self.transform = transform
        self.id_dict = {}

        self.class_names = os.listdir(root)
        self.class_name_id_dict = {c: i for i, c in enumerate(self.class_names)}

        self.images = []
        for class_name in self.class_names:
            image_dir = os.path.join(root, class_name, mode, "good")
            self.images += glob.glob(f"{image_dir}/*.png")

    def __getitem__(self, item):

        image_path = self.images[item]
        image_name = os.path.basename(image_path)
        class_name = image_path.rsplit("/", 4)[-4]
        class_id = self.class_name_id_dict[class_name]

        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)

        return image, class_id, image_name

    def __len__(self):
        return len(self.images)


def eval_dataset(data_dir, train):
    from torchvision import transforms

    train_transformer = transforms.Compose([
        transforms.RandomResizedCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_data = ClassifyDataset(data_dir, mode=train, transform=train_transformer)

    data_ = train_data[1]

    dataloader = data.DataLoader(
        train_data, batch_size=32, num_workers=2, shuffle=False, pin_memory=True,
    )
    for images, label_ids, img_names in dataloader:

        pass


# ----------------------------------------------------
if __name__ == '__main__':
    eval_dataset("/home/log/PycharmProjects/fastflow_v2/datasets/classify_data", "train")