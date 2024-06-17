import glob
import os
from typing import Dict, List, Union

import pandas as pd
from filelock import FileLock
from transformers import is_torch_available

from src.utils.preprocessing import InputExample, get_examples_to_features_fn


def wiki_read_col_data(settings, mode):
    file_path = get_file(settings.data_dir, mode)
    data = pd.read_parquet(file_path)
    labels = data['ner_tags'].tolist()  # 假设您的数据中有一个'label'列
    max_len = settings.symbol_max_seq_length - 1
    for i in range(len(labels)):
        if len(labels[i]) > max_len:
            labels[i] = labels[i][:max_len]
    return labels


if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset


    class WikiDataset(Dataset):

        features: List[Dict[str, Union[int, torch.Tensor]]]

        def __init__(self, data_dir, processor, transforms, modality, max_seq_length, mode):
            # Load data features from cache or dataset file
            cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}".format(mode, processor.__class__.__name__, str(max_seq_length)),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file):
                    self.features = torch.load(cached_features_file)
                else:
                    self.examples = read_examples_from_file(data_dir, mode)
                    examples_to_features_fn = get_examples_to_features_fn(modality)
                    self.features = examples_to_features_fn(
                        examples=self.examples,
                        max_seq_length=max_seq_length,
                        processor=processor,
                        transforms=transforms,
                    )
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i):
            return self.features[i]


def get_file(data_dir, mode):
    if mode == "dev":
        mode = "validation"
    fp = os.path.join(data_dir, f"*{mode}*.parquet")
    _fp = glob.glob(fp)
    if len(_fp) == 1:
        return _fp[0]
    elif len(_fp) == 0:
        return None
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def read_examples_from_file(data_dir, mode):
    file_path = get_file(data_dir, mode)
    examples = []

    data = pd.read_parquet(file_path)
    tokens = data['tokens']
    for i in range(len(tokens)):
        words = tokens[i].tolist()
        examples.append(InputExample(words=words))

    return examples
