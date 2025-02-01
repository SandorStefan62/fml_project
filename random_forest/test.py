import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import get_data_loaders

train_loader, test_loader, _ = get_data_loaders('../dataset/dataset.csv', for_rf=True)