#!/usr/bin/env python3
import logging
import os
import sys

import pandas as pd
import torch
import torchaudio
import transformers
from datasets import Dataset
from packaging import version
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn import functional as F
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
    is_apex_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from arg_parsers import DataTrainingArguments, ModelArguments
from models import Wav2Vec2ClassificationModel
from processors import CustomWav2Vec2Processor

os.environ["WANDB_DISABLED"] = "true"
if is_apex_available():
    from apex import amp


if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets:

    metadata_train_path = "../trainset.csv"
    metadata_val_path = "../valset.csv"
    metadata_test_path = "../testset.csv"

    dataset_folder = "dataset"
    dataset_wav_folder = "dataset_wav"

    train_metadata = pd.read_csv(metadata_train_path)
    val_metadata = pd.read_csv(metadata_val_path)

    train_dataset = Dataset.from_pandas(train_metadata)
    eval_dataset = Dataset.from_pandas(val_metadata)

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16_000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = CustomWav2Vec2Processor(feature_extractor=feature_extractor)
    model = Wav2Vec2ClassificationModel.from_pretrained(
        "bakrianoo/sinai-voice-ar-stt",
        attention_dropout=0.01,
        hidden_dropout=0.01,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.01,
        gradient_checkpointing=True,
        num_attention_heads=4,
    )

    if model_args.freeze_feature_extractor:
        model.freeze_feature_extractor()

    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if data_args.max_val_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    # Preprocessing the datasets.
    # We need to read the aduio files as arrays and tokenize the targets.
    resamplers = {  # The dataset contains all the uncommented sample rates
        48000: torchaudio.transforms.Resample(48000, 16000),
        44100: torchaudio.transforms.Resample(44100, 16000),
        # 32000: torchaudio.transforms.Resample(32000, 16000),
    }

    labels = {
        bahr: bahr_index
        for bahr_index, bahr in enumerate(sorted(set(train_metadata["Bahr"])))
    }
    print("labels are:", labels)
    print("len:", len(labels))

    def speech_file_to_array_fn(batch):
        start = 0
        stop = 20
        srate = 16_000
        speech_array, sampling_rate = torchaudio.load(
            f'../dataset_wav/{batch["Utterance name"]}'
        )
        speech_array = speech_array[0]
        batch["speech"] = resamplers[sampling_rate](speech_array).squeeze().numpy()
        batch["sampling_rate"] = srate
        batch["parent"] = labels[batch["Bahr"]]
        return batch

    train_dataset = train_dataset.map(
        speech_file_to_array_fn,
        remove_columns=train_dataset.column_names,
        num_proc=data_args.preprocessing_num_workers,
    )
    eval_dataset = eval_dataset.map(
        speech_file_to_array_fn,
        remove_columns=eval_dataset.column_names,
        num_proc=data_args.preprocessing_num_workers,
    )

    def prepare_dataset(batch):
        # check that all files have the correct sampling rate
        assert (
            len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
        batch["input_values"] = processor(
            batch["speech"], sampling_rate=batch["sampling_rate"][0]
        ).input_values
        batch["labels"] = batch["parent"]
        return batch

    train_dataset = train_dataset.map(
        prepare_dataset,
        remove_columns=train_dataset.column_names,
        batch_size=training_args.per_device_train_batch_size,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
    )
    eval_dataset = eval_dataset.map(
        prepare_dataset,
        remove_columns=eval_dataset.column_names,
        batch_size=training_args.per_device_train_batch_size,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
    )

    from sklearn.metrics import classification_report, confusion_matrix

    def compute_metrics(pred):
        labels = pred.label_ids.argmax(-1)
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        report = classification_report(labels, preds)
        matrix = confusion_matrix(labels, preds)
        print(matrix)
        return {"accuracy": acc}

    # Data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    # Initialize our Trainer
    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=processor.feature_extractor,
    )
    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        # save the feature_extractor and the tokenizer
        if is_main_process(training_args.local_rank):
            processor.save_pretrained(training_args.output_dir)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_val_samples = (
            data_args.max_val_samples
            if data_args.max_val_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    return results


if __name__ == "__main__":
    main()
