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

        if self.task != "pretrain":
            self.n_labels = len(settings.target_label_switch)
            self.out_fnn = nn.Linear(settings.dim_out, self.n_labels)

        self.model_encoder, self.model_embedding, self.model_config = self.get_encoder()
        self.SSS_embedding = SSSEmbedding(self.settings)

    def get_encoder(self):
        model = AutoModel.from_pretrained(self.settings.model_name_or_path)
        return model.encoder, model.embeddings, model.config

    def forward(self, batch, run_test):
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
            loss_dist = dist_similarity(text_dist, symb_dist)

            symb_spat = get_token(E_g, seq_lengths, symb_word_starts)
            text_spat = get_token(E_t, seq_lengths, text_word_starts).detach()
            loss_spat = spat_similarity(symb_spat.view(-1, symb_spat.shape[-1]),
                                        text_spat.view(-1, text_spat.shape[-1]))

            last_scores = 0

            return last_scores, loss_dist, loss_spat
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
            # Get the final scores
            last_scores = self.out_fnn(prefix_output)
            if not run_test:
                loss_sc = 0
                loss_nce = 0
                return last_scores, loss_sc, loss_nce
            else:
                return last_scores


def prefix_alignment(output, seq_lengths, word_starts):
    max_lengths = min(max(seq_lengths), output.shape[1])
    output_alignment = torch.zeros([len(seq_lengths), max_lengths, output.shape[2]]).to(output.device)
    for b in range(len(seq_lengths)):
        for index, word_ind in enumerate(word_starts[b]):
            if index < max_lengths and word_ind < max_lengths:
                output_alignment[b, index, :] = output[b, word_ind, :]
    return output_alignment


def get_token(output, seq_lengths, word_starts):
    token_list = []
    for b in range(len(seq_lengths)):
        for index, word_ind in enumerate(word_starts[b]):
            if index < max(seq_lengths):
                token_list.append(output[b, word_ind, :])
    return torch.stack(token_list, dim=0)


def get_extended_attention_mask(attention_mask, input_shape, device):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask
