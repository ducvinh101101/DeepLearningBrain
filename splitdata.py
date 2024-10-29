import os
import shutil
from sklearn.model_selection import train_test_split

base_dir = 'dataset'
classes = ['healthy', 'tumor', 'stroke']
for cls in classes:
    os.makedirs(os.path.join(base_dir, f'train/{cls}'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, f'test/{cls}'), exist_ok=True)


def split_data(SOURCE, TRAINING, TEST, train_size, max_files=None):
    all_files = os.listdir(SOURCE)
    all_files = [f for f in all_files if os.path.isfile(os.path.join(SOURCE, f))]
    if max_files:
        all_files = all_files[:max_files]
    train_files, test_files = train_test_split(all_files, train_size=train_size)
    for file_name in train_files:
        shutil.copy(os.path.join(SOURCE, file_name), os.path.join(TRAINING, file_name))
    for file_name in test_files:
        shutil.copy(os.path.join(SOURCE, file_name), os.path.join(TEST, file_name))


split_size_train = 0.7

min_images = min(len(os.listdir(os.path.join(base_dir, cls))) for cls in classes)
for cls in classes:
    SOURCE = os.path.join(base_dir, cls)
    TRAINING = os.path.join(base_dir, 'train', cls)
    TEST = os.path.join(base_dir, 'test', cls)
    split_data(SOURCE, TRAINING, TEST, split_size_train, max_files=min_images)
