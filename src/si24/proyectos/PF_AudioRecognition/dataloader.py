from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

CLASS_LABELS = {
    0: "yes",
    1: "no",
    2: "up",
    3: "down",
    4: "left",
    5: "right",
    6: "on",
    7: "off",
    8: "stop",
    9: "go",
    10: "zero",
    11: "one",
    12: "two",
    13: "three",
    14: "four",
    15: "five",
    16: "six",
    17: "seven",
    18: "eight",
    19: "nine",
    20: "bed",
    21: "bird",
    22: "cat",
    23: "dog",
    24: "happy",
    25: "house",
    26: "marvin",
    27: "sheila",
    28: "tree",
    29: "wow",
    30: "_silence_",
}


class SpeechDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data[idx]), self.target[idx]
        return self.data[idx], self.target[idx]


if __name__ == "__main__":
    dataset = load_dataset("speech_commands", "v0.01")
    train = dataset["train"]
    val = dataset["validation"]

    for i in range(5):
        print(val[i])
