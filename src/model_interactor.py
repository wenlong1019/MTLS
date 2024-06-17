import json
import math

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import MyDataset
from src.loss_functions import cross_entropy_loss
from src.mtls import MTLS
from src.utils.padded_collate import padded_collate


class ModelInteractor:
    def __init__(self, settings):
        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.optimizer = None
        self.scheduler = None

        self.settings = settings
        self.device = settings.device
        self.batch_size = settings.batch_size
        self.step = settings.step

        self.model = MTLS(settings)
        self.model = self.model.to(self.settings.device)

        self._store_settings()
        self._init_data(settings)

    def _init_optimizer(self, stage):
        embeddings_params = []
        encoder_params = []
        other_params = []
        no_train_params = []

        total_params = 0
        trainable_params = 0

        for name, para in self.model.named_parameters():
            total_params += para.numel()
            if "my_encoder" in name:
                if stage == 1:
                    para.requires_grad = False
                elif stage == 2:
                    para.requires_grad = True

            if "textual_embeddings" in name:
                para.requires_grad = False

            if para.requires_grad:
                trainable_params += para.numel()
                if "my_embeddings" in name:
                    embeddings_params += [para]
                elif "my_encoder" in name:
                    encoder_params += [para]
                else:
                    other_params += [para]
            else:
                no_train_params += [para]

        print('total params = {:e}, trainable params = {:e}, Percentage of trainable parameters = {:.4f}%'
              .format(total_params, trainable_params, trainable_params / total_params))

        params = [{"params": embeddings_params, "lr": self.settings.lr_embeddings},
                  {"params": encoder_params, "lr": self.settings.lr_encoder},
                  {"params": other_params, "lr": self.settings.lr_other}, ]

        self.optimizer = torch.optim.AdamW(
            params,
            betas=(self.settings.beta1, self.settings.beta2),
            eps=self.settings.epsilon,
            weight_decay=self.settings.l2)

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)

    def _store_settings(self):
        with open(self.settings.output_dir + "settings.json", "w") as fh:
            json.dump({k: v for k, v in self.settings.__dict__.items() if k not in "device".split()}, fh)

    def _init_data(self, settings):
        if settings.train_file is not None:
            self.train_data = MyDataset(settings.train_file, mode="train", settings=self.settings)
            self.train_loader = DataLoader(self.train_data,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           collate_fn=padded_collate)
        if settings.val_file is not None:
            self.val_data = MyDataset(settings.val_file, mode="dev", settings=self.settings)
            self.val_loader = DataLoader(self.val_data,
                                         batch_size=self.batch_size,
                                         shuffle=False,
                                         collate_fn=padded_collate)
        if settings.test_file is not None:
            self.test_data = MyDataset(settings.test_file, mode="test", settings=self.settings)
            self.test_loader = DataLoader(self.test_data,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          collate_fn=padded_collate)

    def _run_train_batch(self, batch, optimizer, stage):
        optimizer.zero_grad()
        # label [batch x label x head_seq x dependent_seq]
        label_scores, loss_sc, loss_nce = self.model(batch, stage, run_test=False)
        # loss_ce = cross_entropy_loss(label_scores, batch)
        if stage == 1:
            loss_ce = 0
            loss = loss_sc + loss_nce
        else:
            loss_ce = cross_entropy_loss(label_scores, batch)
            loss = loss_ce
        loss.backward()
        loss = float(loss)
        optimizer.step()
        return loss, loss_ce, loss_sc, loss_nce

    def _run_test_batch(self, batch, stage):
        label_scores = self.model(batch, stage, run_test=True)
        predictions = {}
        for i, size in enumerate(batch.seq_lengths):
            size = size.item()
            scores = F.softmax(label_scores[i, 1:size, :], dim=1).cpu()
            prediction = torch.argmax(scores, dim=1).float()
            predictions[batch.graph_ids[i]] = prediction
        return predictions

    def _run_train_epoch(self, data, stage):
        self.model.train()

        total_loss = 0
        total_ce = 0
        total_sc = 0
        total_nce = 0

        for i, batch in enumerate(tqdm(data)):
            batch.to(self.device)
            loss, loss_ce, loss_sc, loss_nce = self._run_train_batch(batch, self.optimizer, stage)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            total_loss += loss
            total_ce += loss_ce
            total_sc += loss_sc
            total_nce += loss_nce

        return total_loss, total_ce, total_sc, total_nce

    def predict(self, data_loader):
        self.model.eval()
        predictions = {}
        for batch in data_loader:
            batch.to(self.device)
            with torch.no_grad():
                pred = self._run_test_batch(batch, stage=2)
                predictions.update(pred)
        return predictions

    def save(self, path):
        state = {"model": self.model.state_dict()}
        torch.save(state, self.settings.output_dir + path)

    def load(self, path, load_all=False):
        print("Restoring model from {}".format(path))
        state = torch.load(path)
        #####################################################################################
        # resize_model_embeddings(self.model.visual_encoder, self.settings.symbol_max_seq_length)
        #####################################################################################
        if not load_all:
            pretrained_model_dict = {}
            for name in state["model"]:
                # if "my_encoder" in name or "my_embeddings" in name:
                if "my_embeddings" in name:
                    pretrained_model_dict[name] = state["model"][name]
            self.model.load_state_dict(pretrained_model_dict, strict=False)
        else:
            self.model.load_state_dict(state["model"])

        self.model = self.model.to(self.settings.device)

    def train_stage1(self):
        stage = 1
        self._init_optimizer(stage)
        settings = self.settings
        print("Training is starting for {} steps using ".format(settings.step) +
              "{} with the following settings:".format(self.device))
        print()
        for key, val in settings.__dict__.items():
            print("{}: {}".format(key, val))
        print(flush=True)
        stage = 1
        ################################################################
        best_loss = 1000000
        best_loss_epoch = 1
        early_stop = False
        ################################################################
        epoch_batch = len(self.train_loader)
        epochs = math.ceil(settings.step / epoch_batch)
        for epoch in range(1, epochs + 1):
            if not early_stop:
                print("#" * 50)
                print("Epoch:{}".format(epoch))
                total_loss, total_ce, total_sc, total_nce = self._run_train_epoch(self.train_loader, stage)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print("loss {:.4f}".format(total_loss))
                print("loss_ce:{:.4f}  loss_sc:{:.4f}  loss_nce:{:.4f}".format(total_ce, total_sc, total_nce))
                print('LR_textual_embedding {}'.format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                print('LR_textual_encoder {}'.format(self.optimizer.state_dict()['param_groups'][1]['lr']))
                print('LR_textual_other {}'.format(self.optimizer.state_dict()['param_groups'][2]['lr']))
                ##################################################################
                # change learning rate
                self.scheduler.step(total_loss)
                ###########################################################
                improvement = total_loss < best_loss
                elapsed = epoch - best_loss_epoch
                if not improvement:
                    print("Have not seen any improvement for {} epochs".format(elapsed))
                    print("Best loss was {:.4f} seen at epoch #{}".format(best_loss, best_loss_epoch))
                    if elapsed == 20:
                        early_stop = True
                        print("!!!!!!!!!!!!!!!!!!!!early_stop!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                else:
                    best_loss = total_loss
                    best_loss_epoch = epoch
                    print("Saving {} model".format(best_loss_epoch))
                    self.save("stage1_best_model.save")
                    print("Best f1 was {:.4f} seen at epoch #{}".format(best_loss, best_loss_epoch))
        self.save("stage1_last_epoch.save")
