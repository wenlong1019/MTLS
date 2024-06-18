import os
import shutil
import sys
from argparse import ArgumentParser

import torch

from src.ner.ner_interactor import NerInteractor
from src.utils.label_type import ner_target_label, ner_target_label_switch
from src.utils.logger import Logger


def get_args(forced_args=None):
    parser = ArgumentParser()

    # Model hyperparameters
    parser.add_argument("--task", type=str, default="ner")
    parser.add_argument("--do_load", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--step", type=int, default=15000)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--text_max_seq_length", type=int, default=512)
    parser.add_argument("--symbol_max_seq_length", type=int, default=529)
    parser.add_argument("--seed", help="Sets the random seed", type=int, default=3407)

    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--num_selected_experts", type=int, default=2)
    parser.add_argument("--temperature", type=int, default=0.2)
    parser.add_argument("--encoder_layers", type=int, default=6)
    parser.add_argument("--decoder_layers", type=int, default=6)
    parser.add_argument("--dropout_rate", type=int, default=0.1)

    # AdamW optimizer hyperparameters
    optimizer = parser.add_argument_group("Optimizer", "Set the AdamW optimizer hyperparameters")
    optimizer.add_argument("--lr_embeddings", type=float, default=5e-5)
    optimizer.add_argument("--lr_encoder", type=float, default=5e-5)
    optimizer.add_argument("--lr_other", type=float, default=5e-5)
    optimizer.add_argument("--beta1", type=float, default=0.9)
    optimizer.add_argument("--beta2", type=float, default=0.999)
    optimizer.add_argument("--epsilon", type=float, default=1e-8)
    optimizer.add_argument("--l2", type=float, default=0.05)

    # ReduceLROnPlateau scheduler hyperparameters
    scheduler = parser.add_argument_group("Scheduler", "Set the ReduceLROnPlateau scheduler hyperparameters")
    scheduler.add_argument("--factor", type=float, default=0.5)
    scheduler.add_argument("--patience", type=float, default=5)
    scheduler.add_argument("--min_lr", type=float, default=1e-7)

    # Files hyperparameters
    parser.add_argument("--model_name_or_path", metavar="FILE")
    parser.add_argument("--load", help="Load trained model", metavar="FILE")
    parser.add_argument("--SSS_embedding_load", metavar="FILE")
    parser.add_argument("--output_dir", type=str, default="./experiments/")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--train_file", metavar="FILE")
    parser.add_argument("--renderer_config_dir", default="./renderer_config")
    parser.add_argument("--fallback_fonts_dir", default="./renderer_config/fallback_fonts/")

    args = parser.parse_args(forced_args)
    return args


def run_task(args):
    # Set random seed and device
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not args.output_dir.endswith("/"):
        args.output_dir += "/"

    args.target_label = ner_target_label
    args.target_label_switch = ner_target_label_switch

    # Initialize model
    model = NerInteractor(args)

    if args.do_load:
        model.load(args.load)
    else:
        print("#" * 50)
        print("#" * 16 + "training!!!" + "#" * 17)
        print("#" * 50)
        model.load(args.SSS_embedding_load, load_all=False)
        model.train()
        print("#" * 50)
        model.load(args.output_dir + "best_model.save", load_all=True)

    if args.val_file:
        model.predict_val()

    if args.test_file:
        model.predict_test()


if __name__ == "__main__":
    args = get_args()
    logger = Logger(sys.stdout)

    # ["ar", "bg", "cs", "el", "en", "fa", "fr",  "ja", 
    # "ko", "ru",  "ta", "tr", "ur", "vi", "zh"]
    for dataset in ["ar", "bg", "cs", "el", "en", "fa", "fr",  "ja",
                    "ko", "ru",  "ta", "tr", "ur", "vi", "zh"]:
        # Model file
        model_name = "bert-base-cased"  # "bert-base-cased" or "roberta-base"
        model_path = "./model/{}".format(model_name)
        args.model_name_or_path = model_path

        args.SSS_embedding_load = "./experiments/pretrain/try/{}/pretrain_best_model.save".format(model_name)
        # args.load = "./experiments/{}/{}/{}/best_model.save".format(args.task, dataset, model_name)\

        # Experiment files
        setup = "{}-{}".format(args.task, dataset)
        args.data_dir = "./datasets/NER/wikiann/{}".format(dataset)
        args.train_file = "./datasets/NER/wikiann/{}/train".format(dataset)
        args.val_file = "./datasets/NER/wikiann/{}/dev".format(dataset)
        args.test_file = "./datasets/NER/wikiann/{}/test".format(dataset)
        args.output_dir = "./experiments/{}/{}".format(args.task, dataset)

        # Logger set
        if args.load is None:
            if os.path.exists(args.output_dir):
                shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        logger.set_log(args.output_dir, args.load)
        sys.stdout = logger

        # Start train
        print("Running {}".format(setup))
        run_task(args)
        print("$" * 50)

        sys.stdout.close()
