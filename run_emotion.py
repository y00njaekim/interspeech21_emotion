#!/usr/bin/env python3
import logging
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

import librosa
from lang_trans import arabic

import soundfile as sf
from model import Wav2Vec2ForCTCnCLS
from transformers.trainer_utils import get_last_checkpoint
import os

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    is_apex_available,
    trainer_utils,
)

import pickle
from torch.utils.data import DataLoader

# Captum Import
from captum.attr import LayerIntegratedGradients, visualization as viz
from tqdm import tqdm # for progress bar

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    verbose_logging: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log verbose messages or not."},
    )
    alpha: Optional[float] = field(
        default=0.1,
        metadata={"help": "loss_cls + alpha * loss_ctc"},
    )
    beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "loss_cls + alpha * loss_ctc + beta * loss_prosody"},
    )
    tokenizer: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer"}
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Project name for wandb logging"},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Run name for wandb logging"},
    )


def configure_logger(model_args: ModelArguments, training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging_level = logging.WARNING
    if model_args.verbose_logging:
        logging_level = logging.DEBUG
    elif trainer_utils.is_main_process(training_args.local_rank):
        logging_level = logging.INFO
    logger.setLevel(logging_level)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: str = field(
        default='emotion', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_split_name: Optional[str] = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    validation_split_name: Optional[str] = field(
        default="validation",
        metadata={
            "help": "The name of the validation data set split to use (via the datasets library). Defaults to 'validation'"
        },
    )
    target_text_column: Optional[str] = field(
        default="text",
        metadata={"help": "Column in the dataset that contains label (target text). Defaults to 'text'"},
    )
    speech_file_column: Optional[str] = field(
        default="file",
        metadata={"help": "Column in the dataset that contains speech file path. Defaults to 'file'"},
    )
    target_feature_extractor_sampling_rate: Optional[bool] = field(
        default=False,
        metadata={"help": "Resample loaded audio to target feature extractor's sampling rate or not."},
    )
    max_duration_in_seconds: Optional[float] = field(
        default=None,
        metadata={"help": "Filters out examples longer than specified. Defaults to no filtering."},
    )
    orthography: Optional[str] = field(
        default="librispeech",
        metadata={
            "help": "Orthography used for normalization and tokenization: 'librispeech' (default), 'timit', or 'buckwalter'."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # select which split as test
    split_id: str = field(
        default='01F', metadata={"help": "iemocap_ + splitid (e.g. 01M, 02F, etc) + train/test.csv"}
    )

    output_file: Optional[str] = field(
        default=None,
        metadata={"help": "Output file."},
    )

    # Interpretation 관련 필드 추가
    do_interpret: bool = field(
        default=False, metadata={"help": "Whether to compute interpretation maps (IG and Attention)."}
    )
    interpretation_output_dir: Optional[str] = field(
        default=None, metadata={"help": "Output directory for interpretation results."}
    )


@dataclass
class Orthography:
    """
    Orthography scheme used for text normalization and tokenization.

    Args:
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to accept lowercase input and lowercase the output when decoding.
        vocab_file (:obj:`str`, `optional`, defaults to :obj:`None`):
            File containing the vocabulary.
        word_delimiter_token (:obj:`str`, `optional`, defaults to :obj:`"|"`):
            The token used for delimiting words; it needs to be in the vocabulary.
        translation_table (:obj:`Dict[str, str]`, `optional`, defaults to :obj:`{}`):
            Table to use with `str.translate()` when preprocessing text (e.g., "-" -> " ").
        words_to_remove (:obj:`Set[str]`, `optional`, defaults to :obj:`set()`):
            Words to remove when preprocessing text (e.g., "sil").
        untransliterator (:obj:`Callable[[str], str]`, `optional`, defaults to :obj:`None`):
            Function that untransliterates text back into native writing system.
    """

    do_lower_case: bool = False
    vocab_file: Optional[str] = None
    word_delimiter_token: Optional[str] = "|"
    translation_table: Optional[Dict[str, str]] = field(default_factory=dict)
    words_to_remove: Optional[Set[str]] = field(default_factory=set)
    untransliterator: Optional[Callable[[str], str]] = None
    tokenizer: Optional[str] = None

    @classmethod
    def from_name(cls, name: str):
        if name == "librispeech":
            return cls()
        if name == "timit":
            return cls(
                do_lower_case=True,
                # break compounds like "quarter-century-old" and replace pauses "--"
                translation_table=str.maketrans({"-": " "}),
            )
        if name == "buckwalter":
            translation_table = {
                "-": " ",  # sometimes used to represent pauses
                "^": "v",  # fixing "tha" in arabic_speech_corpus dataset
            }
            return cls(
                vocab_file=pathlib.Path(__file__).parent.joinpath("vocab/buckwalter.json"),
                word_delimiter_token="/",  # "|" is Arabic letter alef with madda above
                translation_table=str.maketrans(translation_table),
                words_to_remove={"sil"},  # fixing "sil" in arabic_speech_corpus dataset
                untransliterator=arabic.buckwalter.untransliterate,
            )
        if name == "korean":
            return cls(
                do_lower_case=False,
            )
        raise ValueError(f"Unsupported orthography: '{name}'.")

    def preprocess_for_training(self, text: str) -> str:
        # TODO: 한글에 대해 전처리 점검
        # TODO(elgeish) return a pipeline (e.g., from jiwer) instead? Or rely on branch predictor as is
        if len(self.translation_table) > 0:
            text = text.translate(self.translation_table)
        if len(self.words_to_remove) == 0:
            try:
                text = " ".join(text.split())  # clean up whitespaces
            except:
                text = "NULL"
        else:
            text = " ".join(w for w in text.split() if w not in self.words_to_remove)  # and clean up whilespaces
        return text

    def create_processor(self, model_args: ModelArguments) -> Wav2Vec2Processor:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir
        )
        if self.vocab_file:
            tokenizer = Wav2Vec2CTCTokenizer(
                self.vocab_file,
                cache_dir=model_args.cache_dir,
                do_lower_case=self.do_lower_case,
                word_delimiter_token=self.word_delimiter_token,
            )
        else:
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                self.tokenizer,
                cache_dir=model_args.cache_dir,
                do_lower_case=self.do_lower_case,
                word_delimiter_token=self.word_delimiter_token,
                bos_token=None,
                eos_token=None,
            )
        return Wav2Vec2Processor(feature_extractor, tokenizer)


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

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    audio_only = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if self.audio_only is False:
            # CTC 레이블과 감정 레이블 분리
            label_features = [{"input_ids": feature["labels"][:-1]} for feature in features]
            cls_labels = [feature["labels"][-1] for feature in features]

            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    max_length=self.max_length_labels,
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )

            # replace padding with -100 to ignore loss correctly
            ctc_labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            
            # prosody 레이블 처리 - 없는 경우 기본값 사용
            prosody_features = []
            for feature in features:
                if "prosody" in feature:
                    prosody_features.append(torch.tensor(feature["prosody"], dtype=torch.float32))
                else:
                    prosody_features.append(torch.zeros(4))  # 4-dimensional prosody vector
            
            batch["labels"] = (
                ctc_labels,
                torch.tensor(cls_labels),
                torch.stack(prosody_features)
            )

        # 파일 경로 정보 추가 (있는 경우에만)
        if "file" in features[0]:
            if isinstance(features[0]["file"], list):
                batch["file"] = [feature["file"][0] for feature in features]
            else:
                batch["file"] = [feature["file"] for feature in features]

        return batch


class CTCTrainer(Trainer):
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                kwargs = dict(device=self.args.device)
                if self.deepspeed and inputs[k].dtype != torch.int64:
                    kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
                inputs[k] = v.to(**kwargs)

            if k == 'labels': # labels are list of tensor, not tensor, special handle here
                for i in range(len(inputs[k])):
                    kwargs = dict(device=self.args.device)
                    if self.deepspeed and inputs[k][i].dtype != torch.int64:
                        kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
                    inputs[k][i] = inputs[k][i].to(**kwargs)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


def extract_prosody_tail(example, tail_ratio=0.3):
    """음성의 뒷부분에서 prosody 특성을 추출합니다.
    
    Args:
        example: 오디오 데이터가 포함된 예제
        tail_ratio: 뒷부분 비율 (기본값: 0.3 = 30%)
        
    Returns:
        prosody 특성이 추가된 예제
    """
    y, sr = example["speech"], example["sampling_rate"]
    tail_start = int(len(y)*(1-tail_ratio))
    tail = y[tail_start:]

    # pitch (f0) - use librosa.pyin
    f0, _, _ = librosa.pyin(tail, fmin=50, fmax=600)
    f0 = np.nanmedian(f0)  # Hz
    f0 = 0.0 if np.isnan(f0) else np.log(f0)  # log-Hz

    # energy
    rms = librosa.feature.rms(y=tail).mean()
    energy = np.log(rms + 1e-8)

    # duration
    dur = len(tail)/sr  # seconds

    # speaking rate - 임시로 0으로 설정
    spk_rate = 0.0

    example["prosody"] = np.array([f0, energy, dur, spk_rate], dtype=np.float32)
    return example

def prepare_example(example, data_args, target_sr, orthography, vocabulary_text_cleaner, text_updates, audio_only=False):
    """
    단일 오디오 데이터를 로드, 텍스트 데이터를 정규화.

    Args:
        example (dict): 전처리할 예제 데이터
        data_args: 데이터 관련 인자들
        target_sr: 목표 샘플링 레이트
        orthography: 텍스트 정규화를 위한 orthography 객체
        vocabulary_text_cleaner: 텍스트 정제를 위한 정규식 객체
        text_updates: 텍스트 업데이트 기록을 위한 리스트
        audio_only (bool, optional): True로 설정하면 오디오 데이터만 처리

    Returns:
        dict: 전처리된 데이터
    """
    example["speech"], example["sampling_rate"] = librosa.load(example[data_args.speech_file_column], sr=target_sr)
    if data_args.max_duration_in_seconds is not None:
        example["duration_in_seconds"] = len(example["speech"]) / example["sampling_rate"]
    
    # prosody 특성 추출 추가
    example = extract_prosody_tail(example)
    
    if audio_only is False:
        # Normalize and clean up text; order matters!
        updated_text = orthography.preprocess_for_training(example[data_args.target_text_column])
        updated_text = vocabulary_text_cleaner.sub("", updated_text)
        if updated_text != example[data_args.target_text_column]:
            text_updates.append((example[data_args.target_text_column], updated_text))
            example[data_args.target_text_column] = updated_text
    return example

def prepare_dataset(batch, data_args, processor, cls_label_map, audio_only=False):
    """
    주어진 배치를 전처리하여 입력 값과 (옵션) 레이블을 생성.

    Args:
        batch (dict): 전처리할 배치 데이터
        data_args: 데이터 관련 인자들
        processor: 오디오 처리를 위한 프로세서
        cls_label_map: 클래스 레이블 매핑
        audio_only (bool, optional): True로 설정하면 오디오 데이터만 처리

    Returns:
        dict: 전처리된 배치 데이터
    """
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
    
    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
    
    if audio_only is False:
        cls_labels = list(map(lambda e: cls_label_map[e], batch["emotion"]))
        with processor.as_target_processor():
            batch["labels"] = processor(batch[data_args.target_text_column]).input_ids
            
        # prosody 레이블과 감정 레이블을 함께 처리
        for i in range(len(cls_labels)):
            batch["labels"][i] = batch["labels"][i] + [cls_labels[i]]
            
        # prosody 레이블이 없는 경우 기본값으로 설정
        if "prosody" not in batch:
            batch["prosody"] = [np.zeros(4, dtype=np.float32) for _ in range(len(batch["speech"]))]
    
    return batch

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    configure_logger(model_args, training_args)

    # wandb 설정 적용
    if model_args.wandb_project is not None:
        training_args.wandb_project = model_args.wandb_project
    if model_args.wandb_run_name is not None:
        training_args.run_name = model_args.wandb_run_name

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
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

    orthography = Orthography.from_name(data_args.orthography.lower())
    orthography.tokenizer = model_args.tokenizer
    processor = orthography.create_processor(model_args)
    
    # sample_text = "안녕하세요 한국어 테스트입니다"
    # decomposed = orthography._decompose_korean(sample_text)

    # dataset_pickle_path = os.path.join(model_args.cache_dir, f"{data_args.dataset_name}_{data_args.split_id}_dataset.pkl") if model_args.cache_dir else f"{data_args.dataset_name}_{data_args.split_id}_dataset.pkl"

    # if os.path.exists(dataset_pickle_path):
    #     logger.info(f"Pickle 파일({dataset_pickle_path})이 존재합니다. 데이터셋을 pickle에서 불러옵니다.")
    #     with open(dataset_pickle_path, "rb") as f:
    #         train_dataset, val_dataset, cls_label_map = pickle.load(f)
    # else:
    #     if data_args.dataset_name == 'emotion':
    #         train_dataset = datasets.load_dataset('csv', data_files='iemocap/iemocap_' + data_args.split_id + '.train.csv', cache_dir=model_args.cache_dir)['train']
    #         val_dataset = datasets.load_dataset('csv', data_files='iemocap/iemocap_' + data_args.split_id + '.test.csv', cache_dir=model_args.cache_dir)['train']
    #         cls_label_map = {"e0": 0, "e1": 1, "e2": 2, "e3": 3}
    #     elif data_args.dataset_name == 'kemdy19':
    #         train_dataset = datasets.load_dataset('csv', data_files='kemdy19_balanced/kemdy19_' + data_args.split_id + '.train.csv', cache_dir=model_args.cache_dir)['train']
    #         val_dataset = datasets.load_dataset('csv', data_files='kemdy19_balanced/kemdy19_' + data_args.split_id + '.test.csv', cache_dir=model_args.cache_dir)['train']
    #         cls_label_map = {"e0": 0, "e1": 1, "e2": 2, "e3": 3}
        
    #     with open(dataset_pickle_path, "wb") as f:
    #         pickle.dump((train_dataset, val_dataset, cls_label_map), f)
    #     logger.info(f"데이터셋을 pickle 파일({dataset_pickle_path})에 저장하였습니다.")
        
    if data_args.dataset_name == 'emotion':
        train_dataset = datasets.load_dataset('csv', data_files='iemocap/iemocap_' + data_args.split_id + '.train.csv', cache_dir=model_args.cache_dir)['train']
        val_dataset = datasets.load_dataset('csv', data_files='iemocap/iemocap_' + data_args.split_id + '.test.csv', cache_dir=model_args.cache_dir)['train']
        cls_label_map = {"e0": 0, "e1": 1, "e2": 2, "e3": 3}
    elif data_args.dataset_name == 'kemdy19':
        train_dataset = datasets.load_dataset('csv', data_files='kemdy19_balanced/kemdy19_' + data_args.split_id + '.train.csv', cache_dir=model_args.cache_dir)['train']
        val_dataset = datasets.load_dataset('csv', data_files='kemdy19_balanced/kemdy19_' + data_args.split_id + '.test.csv', cache_dir=model_args.cache_dir)['train']
        cls_label_map = {"e0": 0, "e1": 1, "e2": 2, "e3": 3}
        
    # if data_args.dataset_name == 'emotion':
    #     train_dataset = datasets.load_dataset('csv', data_files='iemocap/iemocap_' + data_args.split_id + '.train.csv', cache_dir=model_args.cache_dir)['train']
    #     val_dataset = datasets.load_dataset('csv', data_files='iemocap/iemocap_' + data_args.split_id + '.test.csv', cache_dir=model_args.cache_dir)['train']
    #     cls_label_map = {"e0": 0, "e1": 1, "e2": 2, "e3": 3}
    # elif data_args.dataset_name == 'kemdy19':
    #     train_dataset = datasets.load_dataset('csv', data_files='kemdy19/kemdy19_' + data_args.split_id + '.train.csv', cache_dir=model_args.cache_dir)['train']
    #     val_dataset = datasets.load_dataset('csv', data_files='kemdy19/kemdy19_' + data_args.split_id + '.test.csv', cache_dir=model_args.cache_dir)['train']
    #     cls_label_map = {"e0": 0, "e1": 1, "e2": 2, "e3": 3}

    model = Wav2Vec2ForCTCnCLS.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        gradient_checkpointing=True,
        vocab_size=len(processor.tokenizer),
        cls_len=len(cls_label_map),
        alpha=model_args.alpha,
        beta=model_args.beta,
        ctc_zero_infinity=True
    )
 
    wer_metric = datasets.load_metric("wer")
    target_sr = processor.feature_extractor.sampling_rate if data_args.target_feature_extractor_sampling_rate else None
    vocabulary_chars_str = "".join(t for t in processor.tokenizer.get_vocab().keys() if len(t) == 1)
    vocabulary_text_cleaner = re.compile(  # remove characters not in vocabulary
        f"[^\s{re.escape(vocabulary_chars_str)}]",  # allow space in addition to chars in vocabulary
        flags=re.IGNORECASE if processor.tokenizer.do_lower_case else 0,
    )
    text_updates = []

    if training_args.do_train:
        train_dataset = train_dataset.map(
            lambda x: prepare_example(x, data_args, target_sr, orthography, vocabulary_text_cleaner, text_updates),
            remove_columns=[data_args.speech_file_column]
        )
    if training_args.do_predict:
        val_dataset = val_dataset.map(
            lambda x: prepare_example(x, data_args, target_sr, orthography, vocabulary_text_cleaner, text_updates, audio_only=True),
            fn_kwargs={'audio_only':True}
        )
    elif training_args.do_eval:
        val_dataset = val_dataset.map(
            lambda x: prepare_example(x, data_args, target_sr, orthography, vocabulary_text_cleaner, text_updates),
            remove_columns=[data_args.speech_file_column]
        )

    if data_args.max_duration_in_seconds is not None:
        def filter_by_max_duration(example):
            return example["duration_in_seconds"] <= data_args.max_duration_in_seconds
        if training_args.do_train:
            old_train_size = len(train_dataset)
            train_dataset = train_dataset.filter(filter_by_max_duration, remove_columns=["duration_in_seconds"])
            if len(train_dataset) > old_train_size:
                logger.warning(
                    f"Filtered out {len(train_dataset) - old_train_size} train example(s) with empty text."
                )
        if training_args.do_predict or training_args.do_eval:
            old_val_size = len(val_dataset)
            val_dataset = val_dataset.filter(filter_by_max_duration, remove_columns=["duration_in_seconds"])
            if len(val_dataset) > old_val_size:
                logger.warning(
                    f"Filtered out {len(val_dataset) - old_val_size} validation example(s) with empty text."
                )

    def filter_empty_text(example):
        return len(example[data_args.target_text_column]) > 0
    
    if training_args.do_train:
        old_train_size = len(train_dataset)
        train_dataset = train_dataset.filter(filter_empty_text)
        if len(train_dataset) > old_train_size:
            logger.warning(
                f"Filtered out {len(train_dataset) - old_train_size} train example(s) with empty text."
            )
    if training_args.do_predict or training_args.do_eval:
        old_val_size = len(val_dataset)
        val_dataset = val_dataset.filter(filter_empty_text)
        if len(val_dataset) > old_val_size:
            logger.warning(
                f"Filtered out {len(val_dataset) - old_val_size} validation example(s) with empty text."
            )
    
    logger.warning(f"Updated {len(text_updates)} transcript(s) using '{data_args.orthography}' orthography rules.")
    if logger.isEnabledFor(logging.DEBUG):
        for original_text, updated_text in text_updates:
            logger.debug(f'Updated text: "{original_text}" -> "{updated_text}"')
    text_updates = None

    if training_args.do_train:
        train_dataset = train_dataset.map(
            lambda x: prepare_dataset(x, data_args, processor, cls_label_map),
            batch_size=training_args.per_device_train_batch_size,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )
    if training_args.do_predict:
        val_dataset = val_dataset.map(
            lambda x: prepare_dataset(x, data_args, processor, cls_label_map, audio_only=True),
            batch_size=training_args.per_device_train_batch_size,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )
    elif training_args.do_eval:
        val_dataset = val_dataset.map(
            lambda x: prepare_dataset(x, data_args, processor, cls_label_map),
            batch_size=training_args.per_device_train_batch_size,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    def compute_metrics(pred):
        cls_pred_logits = pred.predictions[1]
        cls_pred_ids = np.argmax(cls_pred_logits, axis=-1)
        total = len(pred.label_ids[1])
        correct = (cls_pred_ids == pred.label_ids[1]).sum().item() # label = (ctc_label, cls_label)

        ctc_pred_logits = pred.predictions[0]
        ctc_pred_ids = np.argmax(ctc_pred_logits, axis=-1)
        pred.label_ids[0][pred.label_ids[0] == -100] = processor.tokenizer.pad_token_id
        ctc_pred_str = processor.batch_decode(ctc_pred_ids)
        # we do not want to group tokens when computing the metrics
        ctc_label_str = processor.batch_decode(pred.label_ids[0], group_tokens=False)
        if logger.isEnabledFor(logging.DEBUG):
            for reference, predicted in zip(ctc_label_str, ctc_pred_str):
                logger.debug(f'reference: "{reference}"')
                logger.debug(f'predicted: "{predicted}"')
                if orthography.untransliterator is not None:
                    logger.debug(f'reference (untransliterated): "{orthography.untransliterator(reference)}"')
                    logger.debug(f'predicted (untransliterated): "{orthography.untransliterator(predicted)}"')

        wer = wer_metric.compute(predictions=ctc_pred_str, references=ctc_label_str)
        return {"acc": correct/total, "wer": wer, "correct": correct, "total": total, "strlen": len(ctc_label_str)}

    if model_args.freeze_feature_extractor:
        model.freeze_feature_extractor()
        
    torch.autograd.set_detect_anomaly(True)
    
    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.feature_extractor,
    )

    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model() 

    if training_args.do_predict:
        logger.info('******* Predict ********')
        data_collator.audio_only=True
        predictions, labels, metrics = trainer.predict(val_dataset, metric_key_prefix="predict")
        logits_ctc, logits_cls, logits_prosody, attention_weights = predictions
        pred_ids = np.argmax(logits_cls, axis=-1)
        pred_probs = F.softmax(torch.from_numpy(logits_cls).float(), dim=-1)
        print(val_dataset)
        with open(data_args.output_file, 'w') as f:
            for i in range(len(pred_ids)):
                f.write(val_dataset[i]['file'].split("/")[-1] + " " + str(len(val_dataset[i]['input_values'])/16000) + " ")
                pred = pred_ids[i]
                f.write(str(pred)+' ')
                for j in range(4):
                    f.write(' ' + str(pred_probs[i][j].item()))
                f.write('\n')
        f.close()

    elif training_args.do_eval:
        predictions, labels, metrics = trainer.predict(val_dataset, metric_key_prefix="eval")
        logits_ctc, logits_cls, logits_prosody, attention_weights = predictions
        pred_ids = np.argmax(logits_cls, axis=-1)
        correct = np.sum(pred_ids == labels[1])
        acc = correct / len(pred_ids)
        print('correct:', correct, ', acc:', acc)

    # Interpretation (IG + Attention) 계산 로직 추가
    if data_args.do_interpret:
        if data_args.interpretation_output_dir is None:
            data_args.interpretation_output_dir = os.path.join(training_args.output_dir, "interpretation_results")
        
        logger.info("********* Interpretation 계산 시작 (IG + Attention) *********")
        os.makedirs(data_args.interpretation_output_dir, exist_ok=True)

        # 모델 로드 (이미 로드되어 있다면 재사용, 아니라면 로드)
        # main 함수 초반에서 모델 로드 로직을 활용하거나 여기서 다시 로드
        model_path = training_args.output_dir
        if os.path.isdir(model_path):
            checkpoint = get_last_checkpoint(model_path)
            if checkpoint:
                model_path = checkpoint
                logger.info(f"체크포인트를 사용합니다: {checkpoint}")
        elif model_args.model_name_or_path and os.path.isdir(model_args.model_name_or_path):
            model_path = model_args.model_name_or_path
        else:
             logger.warning(f"모델 경로가 유효하지 않아 pretrained 모델을 사용합니다: {model_args.model_name_or_path}")
             model_path = model_args.model_name_or_path
             # Note: If loading a model trained elsewhere, ensure cls_len matches

        logger.info(f"해석을 위해 모델 로드 중: {model_path}")
        model = Wav2Vec2ForCTCnCLS.from_pretrained(
            model_path,
            cache_dir=model_args.cache_dir,
            cls_len=len(cls_label_map), # Ensure cls_label_map is defined earlier
            alpha=model_args.alpha,
            ctc_zero_infinity=True
        ).to(training_args.device)
        model.eval()
        logger.info("모델 로드 완료")

        # 데이터셋 로드 (이미 전처리된 val_dataset 사용)
        # val_dataset 에는 'input_values', 'file' 등이 있어야 함
        # 해석을 위해서는 audio_only=False로 전처리된 데이터셋이 필요할 수 있음
        # 만약 val_dataset이 audio_only=True로 생성되었다면, 다시 로드 및 전처리 필요
        # 여기서는 val_dataset이 적절히 준비되었다고 가정
        logger.info("해석을 위한 데이터셋 준비 중...")
        
        # 해석을 위해 데이터셋 다시 로드
        if data_args.dataset_name == 'emotion':
            val_dataset = datasets.load_dataset('csv', data_files='iemocap/iemocap_' + data_args.split_id + '.test.csv', cache_dir=model_args.cache_dir)['train']
        elif data_args.dataset_name == 'kemdy19':
            val_dataset = datasets.load_dataset('csv', data_files='kemdy19_balanced/kemdy19_' + data_args.split_id + '.test.csv', cache_dir=model_args.cache_dir)['train']
        
        # 데이터셋 전처리
        val_dataset = val_dataset.map(
            prepare_example,
            fn_kwargs={'audio_only':True}
        )
        
        if data_args.max_duration_in_seconds is not None:
            val_dataset = val_dataset.filter(
                filter_by_max_duration,
                remove_columns=["duration_in_seconds"]
            )
        
        val_dataset = val_dataset.map(
            prepare_dataset,
            fn_kwargs={'audio_only':True},
            batch_size=training_args.per_device_eval_batch_size,
            batched=True,
            num_proc=data_args.preprocessing_num_workers
        )
        
        # 데이터 로더 생성
        interpret_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
        interpret_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=interpret_collator)

        # Layer Integrated Gradients 설정
        def wrapper_forward(inputs, attention_mask=None):
            outputs = model(
                input_values=inputs,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                labels=None,
                if_ctc=True,
                if_cls=True
            )
            # 감정 분류 로짓만 반환
            return outputs.logits[1]

        lig = LayerIntegratedGradients(wrapper_forward, model.wav2vec2.feature_extractor)

        total_files = len(interpret_dataloader)
        logger.info(f"총 {total_files}개의 파일에 대해 해석을 수행합니다.")

        for i, batch in enumerate(tqdm(interpret_dataloader, desc="Interpreting samples")):
            input_values = batch['input_values'].to(training_args.device)
            attention_mask = torch.ones(input_values.shape, device=training_args.device)

            file_path = batch['file'][0]  # 배치의 첫 번째 파일 경로 사용
            base_filename = os.path.basename(file_path)
            name, _ = os.path.splitext(base_filename)
            
            logger.debug(f"파일 처리 중 ({i+1}/{total_files}): {base_filename}")

            try:
                # 1. 모델 예측 및 Attention 가중치 얻기
                with torch.no_grad():
                    outputs = model(
                        input_values=input_values,
                        attention_mask=attention_mask,
                        output_attentions=True,
                        output_hidden_states=True,
                        return_dict=True,
                        labels=None,
                        if_ctc=True,
                        if_cls=True
                    )
                logits_cls = outputs.logits[1]  # Classification logits
                attention_weights = outputs.logits[2]  # Attention weights
                predicted_class = torch.argmax(logits_cls, dim=-1).item()

                # Attention 가중치 저장
                att_output_filename = os.path.join(data_args.interpretation_output_dir, f"{name}_attention_weights.npy")
                np.save(att_output_filename, attention_weights.squeeze().cpu().numpy())
                logger.debug(f"Attention weights 저장 완료: {att_output_filename}")

                # 2. Layer Integrated Gradients 계산
                baseline = torch.zeros_like(input_values)
                attributions_ig = lig.attribute(
                    inputs=input_values,
                    baselines=baseline,
                    target=predicted_class,
                    additional_forward_args=(attention_mask,),
                    internal_batch_size=1
                )
                
                # IG 결과는 feature_extractor 출력의 shape를 가짐
                ig_map = attributions_ig.abs().sum(dim=1).squeeze()
                ig_map = ig_map.cpu().numpy()

                # IG 결과 저장
                ig_output_filename = os.path.join(data_args.interpretation_output_dir, f"{name}_ig_feature_extractor.npy")
                np.save(ig_output_filename, ig_map)
                logger.debug(f"IG 결과 저장 완료: {ig_output_filename}")

            except Exception as e:
                logger.error(f"{base_filename} 해석 중 오류 발생: {e}", exc_info=True)
                continue
        
        logger.info("********* Interpretation 계산 완료 *********")

if __name__ == "__main__":
    main()
