import re
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from PIL import Image
from transformers import PreTrainedTokenizer

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")


def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()


@dataclass
class InputExample:
    words: List[str]


class Sentence:
    def __init__(self, id, tokens, tokens_full, text):
        self.id = id
        self.tokens = tokens
        self.tokens_full = tokens_full
        self.text = text

    def print_text(self):
        return self.text

    def __repr__(self):
        return "\n".join([f"# sent_id = {self.id}"] + [f"# text = {self.text}"] + [str(t) for k, t in
                                                                                   sorted(self.tokens_full.items(),
                                                                                          key=lambda x: x[0])] + [""])

    def __iter__(self):
        for token in self.tokens:
            yield token

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index]

    def __setitem__(self, index, value):
        self.tokens[index] = value


def get_examples_to_features_fn(modality):
    if modality == "image":
        return convert_examples_to_image_features
    if modality == "text":
        return convert_examples_to_text_features
    else:
        raise ValueError("Modality not supported.")


def convert_examples_to_image_features(examples, max_seq_length, processor, transforms):
    """Loads a data file into a list of `Dict` containing image features"""

    features = []
    for (ex_index, example) in enumerate(examples):
        encoding = processor(example.words)
        image = encoding.symb_values
        num_patches = encoding.num_text_patches
        word_starts = encoding.word_starts

        symb_values = transforms(Image.fromarray(image))
        attention_mask = get_attention_mask(num_patches, seq_length=max_seq_length)

        # sanity check lengths
        assert len(attention_mask) == max_seq_length

        features.append({"symb_values": symb_values,
                         "attention_mask": attention_mask,
                         "word_starts": word_starts})

    return features


def convert_examples_to_text_features(
        examples: List[InputExample],
        max_seq_length: int,
        processor: PreTrainedTokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=0,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        **kwargs,
) -> List[Dict[str, Union[int, torch.Tensor, List[int]]]]:
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = []
        word_starts = [0]
        word_cnt = 1
        for word in example.words:
            word_tokens = processor.tokenize(word)
            if len(word_tokens) == 0:
                word_tokens = ["▁"]
            word_starts.append(word_cnt)
            word_cnt = word_cnt + len(word_tokens)
            tokens.extend(word_tokens)

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = processor.num_special_tokens_to_add() + (1 if sep_token_extra else 0)
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
        tokens += [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            token_type_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = processor.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        else:
            input_ids += [pad_token] * padding_length
            attention_mask += [0 if mask_padding_with_zero else 1] * padding_length

        assert len(word_starts) == len(example.words) + 1
        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "word_starts": word_starts,
            }
        )
    return features


def get_attention_mask(num_text_patches, seq_length):
    """
    Creates an attention mask of size [1, seq_length]
    The mask is 1 where there is text or a [SEP] black patch and 0 everywhere else
    """
    n = min(num_text_patches + 1, seq_length)  # Add 1 for [SEP] token (black patch)
    zeros = torch.zeros(seq_length)
    ones = torch.ones(n)
    zeros[:n] = ones
    return zeros
