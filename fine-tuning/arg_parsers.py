from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch

from processors import CustomWav2Vec2Processor


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to freeze the feature extractor layers of the model."
        },
    )
    attention_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout ratio for the attention probabilities."},
    )
    activation_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The dropout ratio for activations inside the fully connected layer."
        },
    )
    hidden_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    feat_proj_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The dropout probabilitiy for all 1D convolutional layers in feature extractor."
        },
    )
    mask_time_prob: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "Propability of each feature vector along the time axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
            "vectors will be masked along the time axis. This is only relevant if ``apply_spec_augment is True``."
        },
    )
    # gradient_checkpointing: Optional[bool] = field(
    #     default=True,
    #     metadata={
    #         "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
    #     },
    # )
    layerdrop: Optional[float] = field(
        default=0.0, metadata={"help": "The LayerDrop probability."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_split_name: Optional[str] = field(
        default="train+validation",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    chars_to_ignore: List[str] = list_field(
        default=[",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�"],
        metadata={"help": "A list of characters to remove from the transcripts."},
    )


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: CustomWav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = 320000
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 320000
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods

        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch
