# MTLS: Making Texts into Linguistic Symbols

In the field of linguistics, all languages can be considered as symbolic systems, with each language relying on symbolic processes to associate specific symbols with meanings. In any language, there is a one-to-one correspondence between linguistic symbol and meaning. In different languages, universal meanings follow varying rules of symbolization in one-to-one correspondence with symbols. The majority of work overlooks the properties of languages as symbol systems. In this paper, we shift focus to the symbolic properties of language and introduce MTLS: a pre-training method to improve the multilingual capability of models by Making Text into Linguistic Symbols. Initially, we replace the vocabulary in pre-trained language models with mapping relations between linguistic symbols and semantics. Subsequently, universal semantics within the symbolic system serve as bridges, linking symbols from different languages to the embedding space of the model, thereby enabling the model to process linguistic symbols. To evaluate the effectiveness of MTLS, we conducted experiments on multilingual downstream tasks, using BERT and RoBERTa as the backbone respectively. The results indicate that despite just over 12,000 pieces of English data in pre-training, the improvement that MTLS brings to the multilingual capabilities of models is remarkably significant.

# Setup

- **Attention ! ! !  This is a simple version that we will refine as we go along.**

- You need to configure the appropriate datasets, models and training environments according to the README.md file in each directory of the file.

- We will upload the pre-trained MTLS-B weights file in Hugging Face.

# Pre-train

```
python pre_train.py
```

# Train

###### NER TASK

```
python train_ner.py
```

###### POS TASK

```
python train_pos.py
```
