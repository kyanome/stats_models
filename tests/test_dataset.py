from dataset import MNIST
from torch.utils.data import DataLoader

def test_mnist():
    dataset = MNIST()
    BATCH_SIZE = 100
    data_loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    train_features, train_labels = next(iter(data_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    assert train_labels.size()[0] == BATCH_SIZE
   