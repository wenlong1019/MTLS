from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.pretrain.pretrain_preprocessing import PretrainDataset
from src.utils.rendering.pangocairo_renderer import PangoCairoTextRenderer
from src.utils.rendering.rendering_utils import get_transforms


class PretrainDatasets(Dataset):
    def __init__(self, data_path, mode, settings):
        super().__init__()
        # Initialize the index_entries, symb_values, and text_values
        self.settings = settings
        self.symb_values = None
        self.text_values = None
        self.task = settings.task

        print("Loading data from {}".format(data_path))
        self._load_symb_data(mode)
        self._load_text_data(mode)
        print("Done")

    def _load_symb_data(self, mode):
        # Load text renderer when using symbolic modality
        processor = PangoCairoTextRenderer.from_pretrained(self.settings.renderer_config_dir,
                                                           fallback_fonts_dir=self.settings.fallback_fonts_dir)

        # Load dataset
        transforms = get_transforms(
            do_resize=True, size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
        )

        dataset = PretrainDataset(data_dir=self.settings.data_dir,
                                  processor=processor,
                                  transforms=transforms,
                                  modality="image",
                                  max_seq_length=self.settings.symbol_max_seq_length,
                                  mode=mode)

        self.symb_values = dataset.features

    def _load_text_data(self, mode):
        # Load tokenizer when using text modality
        processor = AutoTokenizer.from_pretrained(self.settings.model_name_or_path, use_fast=True)

        # Load dataset
        dataset = PretrainDataset(data_dir=self.settings.data_dir,
                                  processor=processor,
                                  transforms=None,
                                  modality="text",
                                  max_seq_length=self.settings.text_max_seq_length,
                                  mode=mode)
        self.text_values = dataset.features

    def __len__(self):
        return len(self.symb_values)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        symb_data = self.symb_values[idx]
        text_data = self.text_values[idx]
        return (symb_data["symb_values"], symb_data["attention_mask"], symb_data["word_starts"],
                text_data["input_ids"], text_data["attention_mask"], text_data["word_starts"])
