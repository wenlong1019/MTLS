ner_target_label = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    # 'B-DATE': 7,
    # 'I-DATE': 8
}
ner_target_label_switch = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    # 7: 'B-DATE',
    # 8: 'I-DATE'
}

pos_target_label = {
    "ADJ": 0,
    "ADP": 1,
    "ADV": 2,
    "AUX": 3,
    "CCONJ": 4,
    "DET": 5,
    "INTJ": 6,
    "NOUN": 7,
    "NUM": 8,
    "PART": 9,
    "PRON": 10,
    "PROPN": 11,
    "PUNCT": 12,
    "SCONJ": 13,
    "SYM": 14,
    "VERB": 15,
    "X": 16,
}

pos_target_label_switch = {
    0: "ADJ",
    1: "ADP",
    2: "ADV",
    3: "AUX",
    4: "CCONJ",
    5: "DET",
    6: "INTJ",
    7: "NOUN",
    8: "NUM",
    9: "PART",
    10: "PRON",
    11: "PROPN",
    12: "PUNCT",
    13: "SCONJ",
    14: "SYM",
    15: "VERB",
    16: "X",
}
