import torch


class PaddedBatch:
    """Container class for padded data"""

    def __init__(self, graph_ids, seq_lengths, targets,
                 symb_values, symb_attention_mask, symb_word_starts,
                 text_input_ids, text_attention_mask, text_word_starts):
        self.graph_ids = graph_ids
        self.seq_lengths = seq_lengths
        self.targets = targets

        self.symb_values = symb_values
        self.symb_attention_mask = symb_attention_mask
        self.symb_word_starts = symb_word_starts

        self.text_input_ids = text_input_ids
        self.text_attention_mask = text_attention_mask
        self.text_word_starts = text_word_starts

    def to(self, device):
        self.symb_values = torch.stack(self.symb_values).to(device)
        self.symb_attention_mask = torch.stack(self.symb_attention_mask).to(device)
        self.text_input_ids = torch.tensor(self.text_input_ids).to(device)
        self.text_attention_mask = torch.tensor(self.text_attention_mask).to(device)


def padded_collate(batch):
    # Sort batch by the longest sequence desc
    batch.sort(key=lambda sequence: len(sequence[7]), reverse=True)
    graph_ids, targets, \
        symb_values, symb_attention_mask, symb_word_starts, \
        text_input_ids, text_attention_mask, text_word_starts = zip(*batch)
    seq_lengths = torch.LongTensor([len(indices) for indices in symb_word_starts])
    seq_lengths = seq_lengths.clamp(max=len(symb_attention_mask[0]))
    padded = PaddedBatch(graph_ids, seq_lengths, targets,
                         symb_values, symb_attention_mask, symb_word_starts,
                         text_input_ids, text_attention_mask, text_word_starts)

    return padded
