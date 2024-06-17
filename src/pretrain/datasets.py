import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.pos.pos_preprocessing import PosDataset, pos_read_col_data
from src.pretrain.pretrain_preprocessing import PretrainDataset, pretrain_read_col_data
from src.utils.rendering.pangocairo_renderer import PangoCairoTextRenderer
from src.utils.rendering.rendering_utils import get_transforms
from src.wiki.wiki_preprocessing import WikiDataset, wiki_read_col_data


class Datasets(Dataset):
    def __init__(self, data_path, mode, settings):
        super().__init__()
        self.settings = settings
        self.index_entries = None
        self.symb_values = None  # 不包括cls，但在word_starts中计数了cls
        self.textual_values = None  # 包括cls
        self.task = settings.task

        print("Loading data from {}".format(data_path))
        self._load_visual_data(mode)
        self._load_textual_data(mode)
        self._load_data(mode)
        print("Done")

    def _load_visual_data(self, mode):
        # Load text renderer when using image modality and tokenizer when using text modality
        modality = "image"

        processor = PangoCairoTextRenderer.from_pretrained(
            self.settings.renderer_config_dir,
            fallback_fonts_dir=self.settings.fallback_fonts_dir)

        if processor.max_seq_length != self.settings.symbol_max_seq_length:
            processor.max_seq_length = self.settings.symbol_max_seq_length

        # Load dataset
        transforms = get_transforms(
            do_resize=True,
            size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
        )
        train_dataset = self.get_dataset(self.settings.data_dir,
                                         processor, transforms, modality,
                                         self.settings.symbol_max_seq_length, mode)

        self.symb_values = train_dataset.features

    def _load_textual_data(self, mode):
        # Load text renderer when using image modality and tokenizer when using text modality
        modality = "text"

        processor = AutoTokenizer.from_pretrained(
            self.settings.model_name_or_path,
            use_fast=True,
        )

        # Load dataset
        transforms = None
        train_dataset = self.get_dataset(self.settings.data_dir,
                                         processor, transforms, modality,
                                         self.settings.text_max_seq_length, mode)

        self.textual_values = train_dataset.features

    def get_dataset(self, data_dir, processor, transforms, modality, max_seq_length, split):
        if self.task == "pretrain":
            return PretrainDataset(
                data_dir=data_dir,
                processor=processor,
                transforms=transforms,
                modality=modality,
                max_seq_length=max_seq_length,
                mode=split
            )

        elif self.task == "pos":
            return PosDataset(
                data_dir=data_dir,
                processor=processor,
                transforms=transforms,
                modality=modality,
                max_seq_length=max_seq_length,
                mode=split
            )

        elif self.task == "wiki":
            return WikiDataset(
                data_dir=data_dir,
                processor=processor,
                transforms=transforms,
                modality=modality,
                max_seq_length=max_seq_length,
                mode=split
            )

    def _load_data(self, mode):
        self.index_entries = []
        if self.task == "pretrain":
            data = pretrain_read_col_data(self.settings.data_dir, mode)
            for sentence in data:
                self.index_entries.append(IndexEntry(self.settings, sentence))

        elif self.task == "pos":
            data = pos_read_col_data(self.settings.data_dir, mode)
            for sentence in data:
                self.index_entries.append(IndexEntry(self.settings, sentence))

        elif self.task == "wiki":
            self.index_entries = wiki_read_col_data(self.settings, mode)

    def __len__(self):
        return len(self.index_entries)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        entry = self.index_entries[idx]
        visual_data = self.symb_values[idx]
        textual_data = self.textual_values[idx]
        if self.task == "cls":
            id = idx
            targets = entry
        elif self.task == "xcopa":
            id = idx
            targets = entry
        elif self.task == "wiki":
            id = idx
            targets = torch.Tensor(entry.copy())
        else:
            id = entry._id
            targets = torch.Tensor(entry.targets)
        return (id, targets,
                visual_data["symb_values"], visual_data["attention_mask"], visual_data["word_starts"],
                textual_data["input_ids"], textual_data["attention_mask"], textual_data["word_starts"],)


class IndexEntry:
    def __init__(self, settings, sentence):
        self._id = sentence.id
        self.targets = sentence.make_matrix(settings.target_label)
