import copy
import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from typing import Tuple, Union

import numpy as np
from torchvision.transforms import Compose, InterpolationMode, Lambda, Resize, ToTensor
from transformers.file_utils import PushToHubMixin, copy_func

TEXT_RENDERER_NAME = "text_renderer_config.json"

PreTrainedTextRenderer = Union["PyGameTextRenderer", "PangoCairoTextRenderer"]  # noqa: F821


def get_transforms(
        do_resize: bool = True,
        size: Union[int, Tuple[int, int]] = (16, 8464), ):
    # Convert to RGB
    transforms = [Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img)]

    # Optionally, resize to specified size
    if do_resize and size:
        transforms.append(Resize(size=size, interpolation=InterpolationMode.BICUBIC))

    # Tensorize image
    transforms.append(ToTensor())
    return Compose(transforms)


@dataclass
class Encoding:
    """
    Dataclass storing renderer outputs

    Args:
        symb_values (`numpy.ndarray`):
            A 3D numpy array containing the pixel values of a rendered image
        sep_patches (`List[int]`):
            A list containing the starting indices (patch-level) at which black separator patches were inserted in the
            image.
        num_text_patches (`int`):
            The number of patches in the image containing text (excluding the final black sep patch). This value is
            e.g. used to construct an attention mask.
        word_starts (`List[int]`, *optional*, defaults to None):
            A list containing the starting index (patch-level) of every word in the rendered sentence. This value is
            set when rendering texts word-by-word (i.e., when calling a renderer with a list of strings/words).
        offset_mapping (`List[Tuple[int, int]]`, *optional*, defaults to None):
            A list containing `(char_start, char_end)` for each image patch to map between text and rendered image.
        overflowing_patches (`List[Encoding]`, *optional*, defaults to None):
            A list of overflowing patch sequences (of type `Encoding`). Used in sliding window approaches, e.g. for
            question answering.
        sequence_ids (`[List[Optional[int]]`, *optional*, defaults to None):
            A list that can be used to distinguish between sentences in sentence pairs: 0 for sentence_a, 1 for
            sentence_b, and None for special patches.
    """

    symb_values: np.ndarray
    sep_patches: List[int]
    num_text_patches: int
    word_starts: Optional[List[int]] = None
    offset_mapping: Optional[List[Tuple[int, int]]] = None
    overflowing_patches: Optional[List] = None
    sequence_ids: Optional[List[Optional[int]]] = None


class TextRenderingMixin(PushToHubMixin):
    """
    This is a text rendering mixin used to provide saving/loading functionality for text renderers.
    """

    _auto_class = None

    def __init__(self, **kwargs):
        """Set elements of `kwargs` as attributes."""
        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                raise err

    @classmethod
    def from_pretrained(cls, renderer_config_dir, **kwargs):
        text_renderer_dict = cls.get_text_renderer_dict(renderer_config_dir)
        return cls.from_dict(text_renderer_dict, **kwargs)

    @classmethod
    def get_text_renderer_dict(cls, renderer_config_dir):
        renderer_config_dir = str(renderer_config_dir)
        text_renderer_file = os.path.join(renderer_config_dir, TEXT_RENDERER_NAME)
        with open(text_renderer_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        text_renderer_dict = json.loads(text)
        font_file_name = text_renderer_dict.get("font_file")
        text_renderer_dict["font_file"] = os.path.join(renderer_config_dir, font_file_name)
        return text_renderer_dict

    @classmethod
    def from_dict(cls, text_renderer_dict, **kwargs):
        if "fallback_fonts_dir" in kwargs:
            fallback_fonts_dir = kwargs.pop("fallback_fonts_dir")
            text_renderer_dict.update({"fallback_fonts_dir": fallback_fonts_dir})
        text_renderer = cls(**text_renderer_dict)
        return text_renderer

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.
        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this text renderer instance.
        """
        output = copy.deepcopy(self.__getstate__())
        output["text_renderer_type"] = self.__class__.__name__

        return output

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> PreTrainedTextRenderer:
        """
        Instantiates a text renderer of type [`~text_rendering_utils.TextRenderingMixin`] from the path to
        a JSON file of parameters.
        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.
        Returns:
            A text renderer of type [`~text_rendering_utils.TextRenderingMixin`]: The text_Renderer
            object instantiated from that JSON file.
        """
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        text_renderer_dict = json.loads(text)
        return cls(**text_renderer_dict)

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.
        Returns:
            `str`: String containing all the attributes that make up this text_renderer instance in JSON format.
        """
        dictionary = self.to_dict()

        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                dictionary[key] = value.tolist()

        # make sure only the basename of the font file is stored
        font_file = dictionary.pop("font_file", None)
        if font_file is not None:
            dictionary["font_file"] = os.path.basename(font_file)

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.
        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this text_renderer instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def copy_font_file_to_save_dir(self, save_directory: Union[str, os.PathLike]):
        """
        Copy font file from resolved font filepath to save directory.
        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the font file will be saved.
        """
        if not os.path.isdir(save_directory):
            raise EnvironmentError(
                f"{save_directory} does not appear to exist. Please double-check the spelling"
                f"or create the directory if necessary"
            )

        if not os.path.isfile(self.font_file):
            raise EnvironmentError(
                f"{self.font_file} does not appear to exist. Please ensure the attribute is set"
                f"correctly and the font file exists."
            )

        destination_path = shutil.copy(self.font_file, save_directory)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoTextRenderer"):
        """
        Register this class with a given auto class. This should only be used for custom text renderers as the ones
        in the library are already mapped with `AutoTextRenderer`.
        <Tip warning={true}>
        This API is experimental and may have some slight breaking changes in the next releases.
        </Tip>
        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoTextRenderer"`):
                The auto class to register this new text renderer with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class


TextRenderingMixin.push_to_hub = copy_func(TextRenderingMixin.push_to_hub)
TextRenderingMixin.push_to_hub.__doc__ = TextRenderingMixin.push_to_hub.__doc__.format(
    object="text renderer", object_class="AutoTextRenderer", object_files="text renderer file"
)
