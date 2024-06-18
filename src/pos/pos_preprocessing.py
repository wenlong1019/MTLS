import glob
import os
from typing import Dict, List, Union

import numpy as np
from filelock import FileLock
from transformers import is_torch_available

from src.utils.preprocessing import Sentence, normalize, InputExample, get_examples_to_features_fn


class PosSentence(Sentence):
    def __init__(self, id, tokens, tokens_full, text):
        super().__init__(id, tokens, tokens_full, text)
        # Set the id, tokens, tokens_full, and text attributes
        self.id = id
        self.tokens = tokens
        self.tokens_full = tokens_full
        self.text = text

    # Create a matrix with the target label
    def make_matrix(self, target_label):
        # Get the length of the tokens
        n = len(self.tokens)
        # Create a matrix with zeros
        matrix = np.zeros(n)
        try:
            # Iterate through the tokens
            for t in self:
                # Get the token id
                m = t.id
                # Set the matrix value to the target label
                matrix[m - 1] = target_label[t.upos]
        except KeyError:
            pass
        return matrix


class Token:
    def __init__(self, id, form, lemma, upos, xpos, feats,
                 head, deprel, deps, misc, scope=None):
        self.id = int(id)
        self.form = form
        self.norm = normalize(form)
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats
        self.deprel = deprel


def pos_read_col_data(data_dir, mode):
    tokens = []
    tokens_full = {}
    sid = -1
    text = ""
    file_list = get_file(data_dir, mode)
    for file_path in file_list:
        with open(file_path, encoding='utf-8') as fhandle:
            for line in fhandle:
                if line.startswith("# sent_id"):
                    sid = line.split("=")[1].strip()
                elif line.startswith("# text"):
                    text = line.split("= ")[1].strip()
                elif line.startswith("#sid"):
                    sid = line.split()[1].strip()
                elif line.startswith("#"):
                    continue
                elif line == "\n":
                    yield PosSentence(sid, tokens, tokens_full, text)
                    tokens = []
                    tokens_full = {}
                else:
                    try:
                        t = int(line.strip().split("\t")[0])
                    except:
                        continue
                    tokens.append(Token(*line.strip().split("\t")))
                    tokens_full[len(tokens)] = tokens[-1]


if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset


    class PosDataset(Dataset):

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
    '''Read examples from file system'''
    # Get list of files
    file_list = get_file(data_dir, mode)
    # Initialize list of examples
    examples = []

    # Loop through each file in the file list
    for file_path in file_list:
        # Open the file
        with open(file_path, "r", encoding="utf-8") as f:
            # Initialize list of words
            words: List[str] = []
            # Loop through each line in the file
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
