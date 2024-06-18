import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.ner.ner_preprocessing import NerDataset, ner_read_col_data
from src.pos.pos_preprocessing import PosDataset, pos_read_col_data
from src.utils.rendering.pangocairo_renderer import PangoCairoTextRenderer
from src.utils.rendering.rendering_utils import get_transforms


class Datasets(Dataset):
    def __init__(self, data_path, mode, settings):
        super().__init__()
        self.settings = settings
        # Initialize the index_entries, symb_values, and text_values
        self.index_entries = None
        self.symb_values = None
        self.text_values = None
        self.task = settings.task

        print("Loading data from {}".format(data_path))
        self._load_symb_data(mode)
        self._load_text_data(mode)
        self._load_data(mode)
        print("Done")

    def _load_symb_data(self, mode):
        # Load text renderer when using image modality
        processor = PangoCairoTextRenderer.from_pretrained(self.settings.renderer_config_dir,
                                                           fallback_fonts_dir=self.settings.fallback_fonts_dir)

        # Load dataset
        transforms = get_transforms(
            do_resize=True,
            size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
        )

        dataset = self.get_dataset(data_dir=self.settings.data_dir,
                                   processor=processor,
                                   transforms=transforms,
                                   modality="image",
                                   max_seq_length=self.settings.symbol_max_seq_length,
                                   split=mode)

        self.symb_values = dataset.features

    def _load_text_data(self, mode):
        # Load tokenizer when using text modality
        processor = AutoTokenizer.from_pretrained(self.settings.model_name_or_path, use_fast=True)

        # Load dataset
        dataset = self.get_dataset(data_dir=self.settings.data_dir,
                                   processor=processor,
                                   transforms=None,
                                   modality="text",
                                   max_seq_length=self.settings.text_max_seq_length,
                                   split=mode)

        self.text_values = dataset.features

    def get_dataset(self, data_dir, processor, transforms, modality, max_seq_length, split):
        if self.task == "pos":
            return PosDataset(
                data_dir=data_dir,
                processor=processor,
                transforms=transforms,
                modality=modality,
                max_seq_length=max_seq_length,
                mode=split
            )
        elif self.task == "ner":
            return NerDataset(
                data_dir=data_dir,
                processor=processor,
                transforms=transforms,
                modality=modality,
                max_seq_length=max_seq_length,
                mode=split
            )

    def _load_data(self, mode):
        self.index_entries = []
        if self.task == "pos":
            data = pos_read_col_data(self.settings.data_dir, mode)
            for sentence in data:
                self.index_entries.append(IndexEntry(self.settings, sentence))
        elif self.task == "ner":
            self.index_entries = ner_read_col_data(self.settings, mode)

    def __len__(self):
        return len(self.index_entries)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        entry = self.index_entries[idx]
        symb_data = self.symb_values[idx]
        text_data = self.text_values[idx]
        if self.task == "ner":
            id = idx
            targets = torch.Tensor(entry.copy())
        else:
            id = entry._id
            targets = torch.Tensor(entry.targets)
        return (id, targets,
                symb_data["symb_values"], symb_data["attention_mask"], symb_data["word_starts"],
                text_data["input_ids"], text_data["attention_mask"], text_data["word_starts"],)


class IndexEntry:
    def __init__(self, settings, sentence):
        self._id = sentence.id
        self.targets = sentence.make_matrix(settings.target_label)
