"""
The script predicts the 3D structure of proteins using ProstT5, which can be used to fine-tune the model for cath classification.
"""

import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from accelerate import Accelerator
import pandas as pd
import numpy as np
import gc

# Assuming the tokenizer and model are already loaded
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5")
model = T5ForConditionalGeneration.from_pretrained(
    "Rostlab/ProstT5", torch_dtype=torch.float16
)
# model.full() if device == 'cpu' else model.half()
distributed_state = Accelerator()
model.to(distributed_state.device)


def predict_sequence_3d_batched(sequence_examples, output_file, batch_size=16):

    # Function to process a single batch
    def process_batch(batch_sequences):
        min_len = min([len(s) for s in batch_sequences])
        max_len = max([len(s) for s in batch_sequences])

        # Replace rare/ambiguous amino acids by X and introduce white-space between all sequences
        batch_sequences = [
            " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
            for sequence in batch_sequences
        ]

        # Add prefixes for AA to 3Di translation
        batch_sequences = ["<AA2fold> " + s for s in batch_sequences]

        # Tokenize sequences and pad up to the longest sequence in the batch
        ids = tokenizer.batch_encode_plus(
            batch_sequences,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        ).to(distributed_state.device)

        # Generation configuration for "folding" (AA-->3Di)
        gen_kwargs_aa2fold = {
            "do_sample": True,
            "num_beams": 3,
            "top_p": 0.95,
            "temperature": 1.2,
            "top_k": 6,
            "repetition_penalty": 1.2,
        }

        # Translate from AA to 3Di (AA-->3Di)
        with torch.no_grad():
            translations = model.generate(
                ids.input_ids,
                attention_mask=ids.attention_mask,
                max_length=max_len,  # max length of generated text
                min_length=min_len,  # minimum length of the generated text
                early_stopping=True,  # stop early if end-of-text token is generated
                num_return_sequences=1,  # return only a single sequence
                **gen_kwargs_aa2fold,
            )

        # Move translations to CPU and decode
        translations = translations.cpu()
        decoded_translations = tokenizer.batch_decode(
            translations, skip_special_tokens=True
        )

        # Remove white-spaces between tokens and convert to lower-case
        structure_sequences = [
            "".join(ts.split(" ")) for ts in decoded_translations
        ]  # predicted 3Di strings
        structure_sequences = [
            s.lower() for s in structure_sequences
        ]  # convert to lower-case

        # Delete tensors to free memory
        del ids
        del translations
        torch.cuda.empty_cache()
        gc.collect()

        return structure_sequences

    # Process all sequences in batches
    count = 0
    for i in range(0, len(sequence_examples), batch_size):
        batch_sequences = sequence_examples[i : i + batch_size]
        with distributed_state.split_between_processes(batch_sequences) as sequences:
            batch_structure_sequences = process_batch(sequences)
        batch_result_df = pd.DataFrame(
            list(zip(batch_sequences, batch_structure_sequences)),
            columns=["input_sequence", "predicted_seq3d"],
        )

        batch_result_df.to_csv(output_file, mode="a", header=(count == 0), index=False)

        del batch_result_df
        del batch_sequences
        del batch_structure_sequences
        torch.cuda.empty_cache()  # Clear GPU cache
        gc.collect()  # Collect garbage

        count += 1
        if count % 10 == 0:
            print(f"Processed {count*batch_size} sequences")

    return


# Example data loading and preprocessing
data = pd.read_csv("./cath_w_seqs_share.csv", index_col=0)
architecture_names = {
    (1, 10): "Mainly Alpha: Orthogonal Bundle",
    (1, 20): "Mainly Alpha: Up-down Bundle",
    (2, 30): "Mainly Beta: Roll",
    (2, 40): "Mainly Beta: Beta Barrel",
    (2, 60): "Mainly Beta: Sandwich",
    (3, 10): "Alpha Beta: Roll",
    (3, 20): "Alpha Beta: Alpha-Beta Barrel",
    (3, 30): "Alpha Beta: 2-Layer Sandwich",
    (3, 40): "Alpha Beta: 3-Layer(aba) Sandwich",
    (3, 90): "Alpha Beta: Alpha-Beta Complex",
}

sorted_keys = sorted(architecture_names.keys())
label_index_mapping = {key: index for index, key in enumerate(sorted_keys)}

data["label"] = data.apply(
    lambda row: label_index_mapping[(row["class"], row["architecture"])], axis=1
)

columns_to_check = ["sequences", "label"]
data = data.dropna(subset=columns_to_check)
seq3d_df = pd.read_csv("./foldseek_seq3d.csv")
merged_df = pd.merge(data, seq3d_df, left_on="cath_id", right_on="id")
data = merged_df

sequences = data["sequences"].tolist()
sequences_3d = data["seq3d"].tolist()
labels = data["label"].tolist()

from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
df = data.copy()

# Perform the split
for train_idx, val_idx in sgkf.split(
    df["sequences"], df["label"], groups=df["superfamily"]
):
    train_df = df.iloc[train_idx][["sequences", "label", "superfamily", "seq3d"]]
    val_df = df.iloc[val_idx][["sequences", "label", "superfamily", "seq3d"]]
    break  # We only need the first split for train/validation

# Verify no intersection
train_superfamilies = set(train_df["superfamily"])
val_superfamilies = set(val_df["superfamily"])

train_sequences = train_df["sequences"].tolist()
test_sequences = val_df["sequences"].tolist()
train_sequences_3d = train_df["seq3d"].tolist()
test_sequences_3d = val_df["seq3d"].tolist()
train_labels = train_df["label"].tolist()
test_labels = val_df["label"].tolist()

# Predict 3D sequences
predict_sequence_3d_batched(
    train_sequences, f"train_predicted_seq3d_{distributed_state.process_index}.csv"
)
predict_sequence_3d_batched(
    test_sequences, f"test_predicted_seq3d_{distributed_state.process_index}.csv"
)

print("Train and test predictions saved successfully.")
