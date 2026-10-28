import torch
from PIL import Image
from torchvision import transforms

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 224


class ResizeWithPadding:

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img):

        w, h = img.size

        new_h = self.height
        new_w = int(w * (new_h / h))

        img = img.resize((new_w, new_h), Image.BILINEAR)

        if new_w < self.width:
            padded = Image.new("L", (self.width, self.height), 0)
            padded.paste(img, (0, 0))
            img = padded
        else:
            img = img.resize((self.width, self.height), Image.BILINEAR)

        return img


def load_charset(path):
    with open(path, "r") as f:
        return f.read().strip()


class LabelEncoder:

    def __init__(self, charset):
        self.charset = charset
        self.char2idx = {c: i for i, c in enumerate(charset)}
        self.idx2char = {i: c for i, c in enumerate(charset)}

    def decode(self, indices):

        result = []
        prev = -1

        for idx in indices:
            if idx != prev and idx != len(self.charset):
                result.append(self.idx2char[idx])
            prev = idx

        return ''.join(result)


transform = transforms.Compose([
    transforms.Grayscale(),
    ResizeWithPadding(IMAGE_HEIGHT, IMAGE_WIDTH),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.25])
])


def greedy_decode(logits, blank_idx):

    probs = torch.softmax(logits, dim=2)
    indices = torch.argmax(probs, dim=2)

    results = []

    for seq in indices:
        prev = blank_idx
        decoded = []

        for idx in seq:
            idx = idx.item()

            if idx != blank_idx and idx != prev:
                decoded.append(idx)

            prev = idx

        results.append(decoded)

    return results