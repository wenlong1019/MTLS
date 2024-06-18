import os
import shutil
import sys
from argparse import ArgumentParser

import torch

from src.utils.label_type import ner_target_label, ner_target_label_switch
from src.utils.logger import Logger
from src.wiki.wiki_interactor import WikiInteractor


def get_args(forced_args=None):
    parser = ArgumentParser()
    #####################################################################################
    parser.add_argument("--task", type=str, default="wiki")
    parser.add_argument("--do_load", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--step", type=int, default=15000)
    parser.add_argument("--dim_out", type=int, default=768)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--text_max_seq_length", type=int, default=512)
    parser.add_argument("--symbol_max_seq_length", type=int, default=529)
    parser.add_argument("--freeze_layer", type=int, default=6)
    parser.add_argument("--seed", help="Sets the random seed", type=int, default=3407)
    #####################################################################################
    optimizer = parser.add_argument_group("Optimizer", "Set the AdamW optimizer hyperparameters")
    optimizer.add_argument("--lr_embeddings", type=float, default=5e-5)
    optimizer.add_argument("--lr_encoder", type=float, default=5e-5)
    optimizer.add_argument("--lr_other", type=float, default=5e-5)
    optimizer.add_argument("--beta1",
                           help="Tunes the running average of the gradient",
                           type=float,
                           default=0.9
                           )
    optimizer.add_argument("--beta2",
                           help="Tunes the running average of the squared gradient",
                           type=float,
                           default=0.999
                           )
    optimizer.add_argument("--l2",
                           help="Weight decay or l2 regularization",
                           type=float,
                           default=0.05
                           )
    #####################################################################################
    parser.add_argument("--textual_model_name_or_path", metavar="FILE")
    parser.add_argument("--load", help="Load trained model", metavar="FILE")
    parser.add_argument("--output_dir", type=str, default="./experiments/try")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--train_file", metavar="FILE")
    parser.add_argument("--val_file", metavar="FILE")
    parser.add_argument("--test_file", metavar="FILE")
    parser.add_argument("--renderer_config_dir", default="./renderer_config")
    parser.add_argument("--fallback_fonts_dir", type=str, default="./fallback_fonts/")
    ##############################################################################

    args = parser.parse_args(forced_args)

    return args


def run_task(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not args.output_dir.endswith("/"):
        args.output_dir += "/"

    args.target_label = ner_target_label
    args.target_label_switch = ner_target_label_switch

    model = WikiInteractor(args)

    if args.do_load:
        # 测试结果
        model.load(args.load)
    else:
        # 训练模型
        # print("#" * 50)
        # print("#" * 16 + " 训练第一阶段！！！ " + "#" * 17)
        # print("#" * 50)
        # model.train_stage1()
        print("#" * 50)
        print("#" * 16 + "训练第二阶段！！！" + "#" * 17)
        print("#" * 50)
        model.load("./experiments/stage1_best_model.save", load_all=False)
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
    for dataset in ["en", "fr","th"]:
        ###############################################################################
        # encoder = "xlm-roberta-base"
        textual_encoder = "bert-base-cased"
        textual_model_path = "./model/{}".format(textual_encoder)
        args.textual_model_name_or_path = textual_model_path
        # args.load = "./experiments/{}/{}/{}/best_model.save".format(args.task,dataset, textual_encoder)
        ###############################################################################
        setup = "{}-{}".format(args.task, dataset)
        # args.model_name_or_path = "./model/{}".format(encoder)
        args.data_dir = "./datasets/wikiann/{}".format(dataset)
        args.train_file = "./datasets/wikiann/{}/train".format(dataset)
        args.val_file = "./datasets/wikiann/{}/dev".format(dataset)
        args.test_file = "./datasets/wikiann/{}/test".format(dataset)
        args.output_dir = "./experiments/{}/{}".format(args.task, dataset)
        ###############################################################
        if args.load is None:
            if os.path.exists(args.output_dir):
                shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        ###############################################################
        logger.set_log(args.output_dir, args.load)
        sys.stdout = logger
        ###############################################################
        print("Running {}".format(setup))
        run_task(args)
        print("$" * 50)

        sys.stdout.close()