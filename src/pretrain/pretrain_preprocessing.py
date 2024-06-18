import glob
import os
from typing import Dict, List, Union

from filelock import FileLock
from transformers import is_torch_available

from src.utils.preprocessing import InputExample, get_examples_to_features_fn

if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset


    class PretrainDataset(Dataset):

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
                        transforms=transforms)
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i):
            return self.features[i]


def get_file(data_dir, mode):
    fp = os.path.join(data_dir, f"*{mode}*.conllu")
    _fp = glob.glob(fp)
    if len(_fp) >= 1:
        return _fp
    elif len(_fp) == 0:
        return None
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def read_examples_from_file(data_dir, mode):
    file_list = get_file(data_dir, mode)
    examples = []

    for file_path in file_list:
        with open(file_path, "r", encoding="utf-8") as f:
            words: List[str] = []
            for line in f.readlines():
                tok = line.strip().split("\t")
                if len(tok) < 2 or line[0] == "#":
                    if words:
                        examples.append(InputExample(words=words))
                        words = []
                if tok[0].isdigit():
                    word = tok[1]
                    words.append(word)
            if words:
                examples.append(InputExample(words=words))
    return examples
