import torch
from torch import nn
from transformers import AutoModel

from src.loss_functions import spat_similarity, dist_similarity
from src.sss_embedding import SSSEmbedding


class MTLS(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.task = settings.task
        # If the task is not pretraining, initialize the number of labels and the output layer
        if self.task != "pretrain":
            self.n_labels = len(settings.target_label_switch)
            self.out_fnn = nn.Linear(settings.embed_dim, self.n_labels)
        # Get the encoder, embedding, and config
        self.model_encoder, self.model_embedding, self.model_config = self.get_encoder()
        # Initialize the SSSEmbedding
        self.SSS_embedding = SSSEmbedding(self.settings)

    def get_encoder(self):
        model = AutoModel.from_pretrained(self.settings.model_name_or_path)
        return model.encoder, model.embeddings, model.config

    def forward(self, batch):
        seq_lengths = batch.seq_lengths
        symb_values = batch.symb_values
        symb_attention_mask = batch.symb_attention_mask
        symb_word_starts = batch.symb_word_starts
        text_input_ids = batch.text_input_ids
        text_attention_mask = batch.text_attention_mask
        text_word_starts = batch.text_word_starts

        E_h, E_g, symb_attention_mask = self.SSS_embedding(symb_values=symb_values, attention_mask=symb_attention_mask)

        if self.task == "pretrain":
            E_t = self.model_embedding(input_ids=text_input_ids,
                                       position_ids=None,
                                       token_type_ids=None,
                                       inputs_embeds=None,
                                       past_key_values_length=0)

            symb_dist = prefix_alignment(E_h, seq_lengths, symb_word_starts)
            text_dist = prefix_alignment(E_t, seq_lengths, text_word_starts).detach()
            # Calculate the loss between the text and symbol embeddings
            loss_dist = dist_similarity(text_dist, symb_dist)

            symb_spat = get_token(E_g, seq_lengths, symb_word_starts)
            text_spat = get_token(E_t, seq_lengths, text_word_starts).detach()
            # Calculate the loss between the symbol and text embeddings
            loss_spat = spat_similarity(symb_spat.view(-1, symb_spat.shape[-1]),
                                        text_spat.view(-1, text_spat.shape[-1]))

            return loss_dist, loss_spat
        else:
            # Get the extended attention mask
            extended_attention_mask = get_extended_attention_mask(symb_attention_mask,
                                                                  symb_attention_mask.size(),
                                                                  symb_values.device)
            # Set the head mask
            head_mask = [None] * self.model_config.num_hidden_layers
            # Run the model encoder
            model_output = self.model_encoder(E_g,
                                              attention_mask=extended_attention_mask,
                                              head_mask=head_mask,
                                              encoder_attention_mask=None)[0]
            # Align the prefix with the model output
            prefix_output = prefix_alignment(model_output, seq_lengths, symb_word_starts)
            last_scores = self.out_fnn(prefix_output)
            return last_scores


def prefix_alignment(output, seq_lengths, word_starts):
    max_lengths = min(max(seq_lengths), output.shape[1])
    # Initialize the output alignment tensor
    output_alignment = torch.zeros([len(seq_lengths), max_lengths, output.shape[2]]).to(output.device)
    # Iterate through each sequence
    for b in range(len(seq_lengths)):
        # Iterate through each word in the sequence
        for index, word_ind in enumerate(word_starts[b]):
            # If the index is less than the maximum length and the word index is also less than the maximum length,
            # copy the output at the word index to the output alignment
            if index < max_lengths and word_ind < max_lengths:
                output_alignment[b, index, :] = output[b, word_ind, :]
    return output_alignment


# Function to get the token from the output, seq_lengths, and word_starts
def get_token(output, seq_lengths, word_starts):
    # Initialize the token_list
    token_list = []
    for b in range(len(seq_lengths)):
        # Iterate through the word_starts
        for index, word_ind in enumerate(word_starts[b]):
            # If the index is less than the maximum sequence length
            if index < max(seq_lengths):
                # Append the output to the token_list
                token_list.append(output[b, word_ind, :])
    return torch.stack(token_list, dim=0)


# Function to get the extended attention mask
def get_extended_attention_mask(attention_mask, input_shape, device):
    if attention_mask.dim() == 3:
        # Convert the attention_mask to the extended attention mask
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Convert the attention_mask to the extended attention mask
        extended_attention_mask = attention_mask[:, None, None, :]
    # If the attention_mask has the wrong shape
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )
    # Convert the extended attention mask to fp16 compatibility
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)  # fp16 compatibility
    # Set the extended attention mask to the appropriate value
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask
