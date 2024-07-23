"""
this script is used to finetune the prostT5 model in a distributed setting on multiple GPUs efficiently
"""

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import datasets
import torch
from transformers import (
    T5Tokenizer,
    T5EncoderModel,
    T5ForSequenceClassification,
    T5PreTrainedModel,
    PretrainedConfig,
    T5Config,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import re
import torch.nn as nn
import evaluate
import numpy as np
from torch.utils.data.dataloader import default_collate
from accelerate import Accelerator
import wandb


class T5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: T5Config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class CustomT5ForSequenceClassification(T5PreTrainedModel):
    def __init__(
        self,
        model_checkpoint,
        config,
        num_labels,
        use_sequence=True,
        use_foldseek=True,
        use_prostT5=True,
        combine_method="average",
    ):

        config.update({"num_labels": num_labels, "classifier_dropout": 0.1})

        super().__init__(config)

        self.transformer = T5EncoderModel.from_pretrained(model_checkpoint)

        if combine_method == "concat":
            config.update({"d_model": 2048})

        self.classification_head = T5ClassificationHead(config)

        self.use_sequence = use_sequence
        self.use_foldseek = use_foldseek
        self.use_prostT5 = use_prostT5
        self.combine_method = combine_method

    def forward(
        self,
        input_ids_sequence=None,
        input_ids_structure_foldseek=None,
        input_ids_structure_prostT5=None,
        attention_mask_sequence=None,
        attention_mask_structure_foldseek=None,
        attention_mask_structure_prostT5=None,
        labels=None,
    ):

        embeddings = []

        # Get embeddings for the sequence
        if self.use_sequence:
            sequence_outputs = self.transformer(
                input_ids_sequence, attention_mask=attention_mask_sequence
            )
            sequence_embeddings = sequence_outputs.last_hidden_state.mean(dim=1)
            embeddings.append(sequence_embeddings)

        if self.use_foldseek:
            # Get embeddings for the structure
            structure_outputs_foldseek = self.transformer(
                input_ids_structure_foldseek,
                attention_mask=attention_mask_structure_foldseek,
            )
            structure_embeddings_foldseek = (
                structure_outputs_foldseek.last_hidden_state.mean(dim=1)
            )
            embeddings.append(structure_embeddings_foldseek)

        if self.use_prostT5:
            structure_outputs_prostT5 = self.transformer(
                input_ids_structure_prostT5,
                attention_mask=attention_mask_structure_prostT5,
            )
            structure_embeddings_prostT5 = (
                structure_outputs_prostT5.last_hidden_state.mean(dim=1)
            )
            embeddings.append(structure_embeddings_prostT5)

        if self.combine_method == "average":
            combined_embeddings = torch.stack(embeddings).mean(dim=0)
        elif self.combine_method == "concat":
            combined_embeddings = torch.cat(embeddings, dim=-1)
        else:
            raise ValueError("Unsupported combine method")

        # Combine the embeddings
        # structure_embeddings = (
        #     structure_embeddings_foldseek + structure_embeddings_prostT5
        # ) / 2.0

        # # Combine the embeddings
        # combined_embeddings = (sequence_embeddings + structure_embeddings) / 2.0

        # Feed to classifier head
        logits = self.classification_head(combined_embeddings)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return (loss, logits) if loss is not None else logits


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(
        self, hf_dataset, use_sequence=True, use_foldseek=True, use_prostT5=True
    ):
        self.dataset = hf_dataset
        self.use_sequence = use_sequence
        self.use_foldseek = use_foldseek
        self.use_prostT5 = use_prostT5

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        result = {}
        if self.use_sequence:
            result["input_ids_sequence"] = torch.tensor(item["input_ids_sequence"])
            result["attention_mask_sequence"] = torch.tensor(
                item["attention_mask_sequence"]
            )
        if self.use_foldseek:
            result["input_ids_structure_foldseek"] = torch.tensor(
                item["input_ids_structure_foldseek"]
            )
            result["attention_mask_structure_foldseek"] = torch.tensor(
                item["attention_mask_structure_foldseek"]
            )
        if self.use_prostT5:
            result["input_ids_structure_prostT5"] = torch.tensor(
                item["input_ids_structure_prostT5"]
            )
            result["attention_mask_structure_prostT5"] = torch.tensor(
                item["attention_mask_structure_prostT5"]
            )
        result["labels"] = torch.tensor(item["labels"])
        return result


def create_dataset(
    tokenized_sequences,
    tokenized_structures_foldseek,
    tokenized_structures_prostT5,
    labels,
):
    input_ids_sequence = [item["input_ids"].squeeze() for item in tokenized_sequences]
    attention_mask_sequence = [
        item["attention_mask"].squeeze() for item in tokenized_sequences
    ]
    input_ids_structure_foldseek = [
        item["input_ids"].squeeze() for item in tokenized_structures_foldseek
    ]
    attention_mask_structure_foldseek = [
        item["attention_mask"].squeeze() for item in tokenized_structures_foldseek
    ]
    input_ids_structure_prostT5 = [
        item["input_ids"].squeeze() for item in tokenized_structures_prostT5
    ]
    attention_mask_structure_prostT5 = [
        item["attention_mask"].squeeze() for item in tokenized_structures_prostT5
    ]

    dataset_dict = {
        "input_ids_sequence": input_ids_sequence,
        "attention_mask_sequence": attention_mask_sequence,
        "input_ids_structure_foldseek": input_ids_structure_foldseek,
        "attention_mask_structure_foldseek": attention_mask_structure_foldseek,
        "input_ids_structure_prostT5": input_ids_structure_prostT5,
        "attention_mask_structure_prostT5": attention_mask_structure_prostT5,
        "labels": labels,
    }

    return Dataset.from_dict(dataset_dict)


def preprocess_data(sequences, structures_foldseek, structures_prostT5, tokenizer):
    tokenized_sequences = []
    tokenized_structures_foldseek = []
    tokenized_structures_prostT5 = []

    for sequence, structures_foldseek, structures_prostT5 in zip(
        sequences, structures_foldseek, structures_prostT5
    ):
        # Preprocess sequences
        sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
        structures_foldseek = " ".join(list(structures_foldseek))
        structures_prostT5 = " ".join(list(structures_prostT5))

        sequence = "<AA2fold> " + sequence if sequence.isupper() else sequence
        structures_foldseek = "<fold2AA> " + structures_foldseek
        structures_prostT5 = "<fold2AA> " + structures_prostT5

        # Tokenize sequences and structures
        sequence_inputs = tokenizer(
            sequence, add_special_tokens=True, padding="longest", return_tensors="pt"
        )
        structure_inputs_foldseek = tokenizer(
            structures_foldseek,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )
        structure_inputs_prostT5 = tokenizer(
            structures_prostT5,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )

        tokenized_sequences.append(sequence_inputs)
        tokenized_structures_foldseek.append(structure_inputs_foldseek)
        tokenized_structures_prostT5.append(structure_inputs_prostT5)

    return (
        tokenized_sequences,
        tokenized_structures_foldseek,
        tokenized_structures_prostT5,
    )


def compute_metrics(eval_preds):
    metric = evaluate.combine(["f1", "precision", "recall"])
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    metrics = metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )
    return metrics


def custom_collate_fn(batch):
    batch_dict = {}

    if "input_ids_sequence" in batch[0]:
        input_ids_sequence = [item["input_ids_sequence"] for item in batch]
        attention_mask_sequence = [item["attention_mask_sequence"] for item in batch]
        input_ids_sequence_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids_sequence, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_mask_sequence_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask_sequence, batch_first=True, padding_value=0
        )
        batch_dict["input_ids_sequence"] = input_ids_sequence_padded
        batch_dict["attention_mask_sequence"] = attention_mask_sequence_padded

    if "input_ids_structure_foldseek" in batch[0]:
        input_ids_structure_foldseek = [
            item["input_ids_structure_foldseek"] for item in batch
        ]
        attention_mask_structure_foldseek = [
            item["attention_mask_structure_foldseek"] for item in batch
        ]
        input_ids_structure_foldseek_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids_structure_foldseek,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        attention_mask_structure_foldseek_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask_structure_foldseek, batch_first=True, padding_value=0
        )
        batch_dict["input_ids_structure_foldseek"] = input_ids_structure_foldseek_padded
        batch_dict["attention_mask_structure_foldseek"] = (
            attention_mask_structure_foldseek_padded
        )

    if "input_ids_structure_prostT5" in batch[0]:
        input_ids_structure_prostT5 = [
            item["input_ids_structure_prostT5"] for item in batch
        ]
        attention_mask_structure_prostT5 = [
            item["attention_mask_structure_prostT5"] for item in batch
        ]
        input_ids_structure_prostT5_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids_structure_prostT5,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        attention_mask_structure_prostT5_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask_structure_prostT5, batch_first=True, padding_value=0
        )
        batch_dict["input_ids_structure_prostT5"] = input_ids_structure_prostT5_padded
        batch_dict["attention_mask_structure_prostT5"] = (
            attention_mask_structure_prostT5_padded
        )

    labels = [item["labels"] for item in batch]
    batch_dict["labels"] = torch.stack(labels)

    return batch_dict


model_checkpoint = "Rostlab/ProstT5"

val_df = pd.read_csv("./val.csv")
train_df = pd.read_csv("./train.csv")

train_sequences = train_df["sequences"].tolist()
val_sequences = val_df["sequences"].tolist()
train_sequences_3d = train_df["seq3d"].tolist()  # from foldseek
val_sequences_3d = val_df["seq3d"].tolist()  # from foldseek
train_sequences_3d_pred = train_df["predicted_seq3d"].tolist()  # from prostT5
val_sequences_3d_pred = val_df["predicted_seq3d"].tolist()  # from prostT5
train_labels = train_df["label"].tolist()
val_labels = val_df["label"].tolist()


tokenizer = T5Tokenizer.from_pretrained(model_checkpoint, do_lower_case=False)

# Preprocess and tokenize the data
(
    train_tokenized_sequences,
    train_tokenized_structures_foldseek,
    train_tokenized_structures_prostT5,
) = preprocess_data(
    train_sequences, train_sequences_3d, train_sequences_3d_pred, tokenizer
)
(
    val_tokenized_sequences,
    val_tokenized_structures_foldseek,
    val_tokenized_structures_prostT5,
) = preprocess_data(val_sequences, val_sequences_3d, val_sequences_3d_pred, tokenizer)


# Create Dataset objects
train_dataset = create_dataset(
    train_tokenized_sequences,
    train_tokenized_structures_foldseek,
    train_tokenized_structures_prostT5,
    train_labels,
)
val_dataset = create_dataset(
    val_tokenized_sequences,
    val_tokenized_structures_foldseek,
    val_tokenized_structures_prostT5,
    val_labels,
)


version = 8
batch_size = 16
train_epochs = 100
num_workers = 8
lr = 1e-5  # 1e-5 for full training
use_sequence = True
use_foldseek = True
use_prostT5 = False
combine_method = "average"


# Create custom dataset
train_dataset = ProteinDataset(
    train_dataset,
    use_sequence=use_sequence,  # Set to False if not using sequences
    use_foldseek=use_foldseek,  # Set to False if not using foldseek
    use_prostT5=use_prostT5,
)
val_dataset = ProteinDataset(
    val_dataset,
    use_sequence=use_sequence,  # Set to False if not using sequences
    use_foldseek=use_foldseek,  # Set to False if not using foldseek
    use_prostT5=use_prostT5,
)


preconfig = PretrainedConfig.from_pretrained(model_checkpoint)
num_labels = max(train_labels + val_labels) + 1

accelerator = Accelerator()

model = CustomT5ForSequenceClassification(
    model_checkpoint,
    preconfig,
    num_labels,
    use_sequence=use_sequence,  # Set to False if not using sequences
    use_foldseek=use_foldseek,  # Set to False if not using foldseek
    use_prostT5=use_prostT5,  # Set to False if not using prostT5
    combine_method=combine_method,
).to(accelerator.device)

import os

os.environ["WANDB_PROJECT"] = "medium biosciences"
os.environ["WANDB_LOG_MODEL"] = "end"
os.environ["WANDB_WATCH"] = "all"
os.environ["WANDB_SILENT"] = "true"


early_stopping = EarlyStoppingCallback(early_stopping_patience=5)

model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned",
    evaluation_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=train_epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    push_to_hub=False,
    fp16=True,
    fp16_full_eval=True,
    # bf16_full_eval=True,
    # bf16=True,
    save_total_limit=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch",
    report_to="wandb",
    lr_scheduler_type="cosine",
    warmup_ratio=0.01,
    logging_strategy="epoch",
    run_name=f"{model_checkpoint.split('/')[-1]}-v-{version}",
    dataloader_num_workers=num_workers,  # accelerator.num_processes*2,
)


trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
    data_collator=custom_collate_fn,
)

trainer.train()
wandb.finish()
