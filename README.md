
# Protein Domain Cath Classification

## Introduction
The task is the classification of the protein domain's CATH architecture. As there has been a lot of development in the field of AI and representation learning, i.e. LLMs and transformers, a lot of these advancements in the field were applied across different domains incl. bio-sciences and what is related to protein data and analysis and studies. As proteins form sequences of amino acids, they can be represented similarly to natural language. Even more, there are similarities at abstract levels such as protein's motif and domain constructs being analogous to words and phrases, which are repeated in different contexts and reused in new sequences. Proteins are more than just 1D sequences, as their functionalities and concrete proporties are much related to their 3D shape and folding etc. However, even their functionalities and 3D shapes are believed to be a consequence of how the amino acids are sequentially ordered in a very complex folded order.

Given that context, researchers are applied the advancements of transformers and LLMs to the protein-related sciences to build powerful models that could model any protein in both sequence- and structure-based forms. Such powerful self-supervised models can be seen as zero-shot learners and used across downstream tasks of interest such as sequence classification. This is simnply done by using a small dataset of proteins represented by their sequences and finetuning a protein-based LLM on this data to learn the task properly. Two examples of such models are prot-bert and ESM_2. On top of that, if the 3D structure information of proteins is given, we could leverage that info as well, in that we could fuse the sequence-based and structure-based embeddings into one embedding (average/concat) and then feed it to the classification layer during the finetuning. An example for that model is ProstT5. 


## Methods
### Data
Using the latest release of the CATH protein classification dataset, I experiment with ProstT5, that can embed both the raw sequences of proteins plus the 3D structure-based sequences, for finetuning the model on the data to learn the classification task. 'eda.ipynb' shows how to download, prepare and split the data.

I generated 1D vector representations of the 3D structural protein information using foldseek, which is used by ProstT5 as input to represent and embed structures of proteins. In addition, I generated predictions of the 3D structures using prostT5 and the protein sequences as inputs to check which structural info serves this task the best foldseek or prostT5 predictions! For more information on that, please refer to the original repo of [[ProstT5](https://github.com/mheinzinger/ProstT5)]

### Models
List of the large language models (LLMs) used for the experiments: Using sequence-only inputs, I used prot-bert and ESM-2 (650M and 35M parameters). Using both sequences and structures, I used ProstT5.

### Experimental Setup
[[Here](https://wandb.ai/shahdsaf/protein-cath-classification-subset?nw=nwusershahdsaf)] you can find several experiments I did on a subset of the dataset I provide in this repo. The first model family I experimented with is ESM-2 and prot-bert, which uses only protein sequences as inputs, while tuning several hyperparameters: lr, batch size, model size, LoRa, freezing the core of the model, and more. I was able finally to reach a validation F1-score of around 0.66 (exp. name on wandb: esm2_t33_650M_UR50D-v-1). Increasing the model size with finetuning the full set of parameters rendered the best results across ESM-2 experiments. 

The second model I experimented with is ProstT5, which uses both protein sequences and structures as inputs, while tuning several hyperparameters: lr, batch size, model size, LoRa, freezing the core of the model, and more. I was able finally to reach a validation F1-score of around 0.7786 (exp. name on wandb: ProstT5-v-11). I tried out both averaging the sequence and structure embeddings or concatenating them but there was not a significant improvement over the averaging. Also, I tried to include the prostT5 structure predictions instead of foldseek vectors but no improvement. Furthermore, I experitmented with different parameters such as higher dropout values, freezing the encoder, and more but there was not big difference. All these trials can be differentiated in the "description" section of each experiment on WandB. One observation is that enforcing higher dropout values (exp. name: ProstT5-v-19) rendered more stable, less overfitting learning curves (but I did not have the time to finish the training and upload the final checkpoint, which might be slightly better than the provided one).

[[Here](https://wandb.ai/shahdsaf/protein%20cath%20classification?nw=nwusershahdsaf)] you can find the most important and final experiments I did on the full training set provided in 'data/train.csv' using the best hyperparameters for prostT5. There are 3 runs v2, v3, v4 corresponding to the usage of protein sequences only, protein 3D foldseek sequences only and the average among both types of sequences respectively. I found out that leveraging both representations retrieve the best validation results on classifying the protein cath classes.

## Discussion
Although the solution leverages both sequence and structure protein information and renders a good perforamance measure on the validation set, there is for sure a room for improvement given the limited time given for this assignment. One of the ideas to try out is to augment the training dataset with synthetic data or data augmentation techniques applied specifically to protein classification tasks.
