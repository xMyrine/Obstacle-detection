from torch.utils.data import Dataset

class HuggingFaceDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.classes = hf_dataset.features['labels'].names

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image']
        image_tensor = self.transform(image)
        label = item['labels']
        return image_tensor, label