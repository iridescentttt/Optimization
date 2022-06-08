from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        self.x = x.float()
        self.y = y.long()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class DataCenter(object):

    def __init__(self, mode, device):
        """dataCenter参数"""
        super(DataCenter, self).__init__()
        self.device = device
        self.mode = mode

    def load_dataSet(self, batch_size):
        """划分数据集"""
        transformation = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(
            './data/', train=True, transform=transformation, download=True)
        test_dataset = datasets.MNIST(
            './data/', train=False, transform=transformation, download=True)

        if self.mode == "GD":
            batch_size = len(train_dataset.targets.numpy())

        train_data = MyDataset(train_dataset.data, train_dataset.targets)
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True)

        test_data = MyDataset(test_dataset.data, test_dataset.targets)
        test_loader = DataLoader(
            test_data, batch_size=128, shuffle=True)

        setattr(self, "train_loader", train_loader)
        setattr(self, "test_loader", test_loader)
        setattr(self, "in_feats",
                train_dataset.data[0].shape[0]*train_dataset.data[0].shape[1])
        setattr(self, "num_classes", len(set(train_dataset.targets.numpy())))
        setattr(self, "total_num", len(train_dataset.targets.numpy()))
