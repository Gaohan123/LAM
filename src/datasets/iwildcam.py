import json
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import numpy as np

val_id_indices=[0, 1, 100, 102, 106, 112, 113, 114, 115, 116, 117, 12, 120, 137, 144, 145, 146, 147, 148, 149, 15, 151, 152, 153, 155, 156, 165, 17, 177, 178, 2, 20, 21, 24, 27, 28, 3, 30, 32, 33, 36, 37, 39, 4, 42, 43, 44, 47, 48, 49, 5, 52, 59, 6, 65, 67, 68, 69, 70, 72, 73, 8, 81, 82, 87, 89, 9, 93, 96, 97, 98]
val_indices=[0, 1, 102, 106, 107, 11, 112, 113, 114, 115, 116, 117, 118, 119, 12, 120, 133, 137, 145, 146, 147, 148, 149, 15, 150, 151, 152, 153, 155, 157, 158, 159, 164, 166, 172, 173, 174, 175, 176, 179, 2, 20, 23, 24, 32, 33, 36, 37, 39, 4, 41, 43, 44, 46, 48, 49, 52, 6, 64, 65, 68, 70, 72, 73, 74, 77, 78, 8, 85, 87, 89, 9, 90, 93, 96]
test_id_indices=[0, 1, 102, 103, 109, 112, 113, 114, 115, 116, 117, 118, 119, 12, 120, 125, 132, 137, 142, 144, 145, 146, 147, 148, 149, 15, 150, 151, 152, 153, 154, 156, 157, 168, 17, 171, 172, 173, 177, 178, 18, 2, 20, 21, 23, 24, 25, 26, 27, 28, 32, 33, 36, 37, 4, 41, 43, 44, 45, 47, 48, 49, 52, 53, 6, 60, 64, 65, 67, 68, 69, 70, 71, 72, 73, 74, 75, 8, 82, 87, 89, 9, 93, 95, 96, 97, 99]
test_indices=[0, 1, 10, 100, 102, 106, 107, 108, 11, 112, 113, 114, 115, 116, 117, 118, 119, 12, 120, 126, 127, 13, 133, 135, 144, 145, 146, 147, 148, 149, 15, 150, 151, 152, 153, 155, 158, 16, 17, 171, 172, 173, 176, 177, 18, 2, 20, 21, 23, 24, 25, 26, 27, 28, 3, 32, 33, 35, 36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 46, 47, 48, 49, 5, 52, 59, 6, 61, 65, 67, 68, 69, 7, 70, 71, 72, 73, 74, 77, 79, 8, 83, 85, 87, 88, 89, 9, 90, 93, 95, 96, 97, 98]



class IWILDCAM(Dataset):
    def __init__(self, root, split='train', name=None, num_examples=None, transform=None, seed=0):
        super().__init__()
        self._split = split
        self._num_examples = num_examples
        self._transform = transform
        self._name = name
        self.data = datasets.ImageFolder(root=root + '/' + split, transform=None)

        if self._num_examples is not None:
            if self._num_examples > len(self.data):
                raise ValueError('num_examples can be at most the dataset size {len(self.data)}')
            rng = np.random.RandomState(seed=seed)
            self._data_indices = rng.permutation(len(self.data))[:num_examples]

    def __getitem__(self, i):
        if self._num_examples is not None:
            i = self._data_indices[i]

        x, y = self.data[i]

        if type(x) is not np.ndarray:
            x = x.convert('RGB')

        if self._transform is not None:
            x = self._transform(x)

        # test_indices
        if self._split == 'test':
            y = test_indices[y]
        # test_id_indices
        elif self._split == 'id_test':
            y = test_id_indices[y]
        # val_indices
        elif self._split == 'val':
            y = val_indices[y]
        # val_id_indices
        elif self._split == 'id_val':
            y = val_id_indices[y]

        return x, y

    def __len__(self) -> int:
        return len(self.data) if self._num_examples is None else self._num_examples
