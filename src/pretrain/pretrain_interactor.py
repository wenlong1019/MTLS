import json
import math

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.mtls import MTLS
from src.pretrain.pretrain_datasets import PretrainDatasets
from src.utils.padded_collate import padded_collate_for_pretrain


class PretrainInteractor:
    def __init__(self, settings):
        # Initialize the training data, data loader, optimizer, scheduler, and MTLS
        self.train_data = None
        self.train_loader = None

        self.optimizer = None
        self.scheduler = None

        self.settings = settings
        self.device = settings.device
        self.batch_size = settings.batch_size
        self.step = settings.step

        self.MTLS = MTLS(settings)
        self.MTLS = self.MTLS.to(self.settings.device)

        # Initialize the training data
        self._init_data(settings)
        self._init_optimizer()
        self._store_settings()

    def _init_data(self, settings):
        self.train_data = PretrainDatasets(settings.train_file, mode="train", settings=self.settings)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       collate_fn=padded_collate_for_pretrain)

    def _init_optimizer(self):
        # Initialize empty lists to store the parameters of the encoder, embedding, and SSS_embedding
        model_encoder_params = []
        model_embedding_params = []
        SSS_embedding_params = []

        # Initialize the total and trainable parameter counts
        total_params = 0
        trainable_params = 0

        # Iterate through the parameters of the model
        for name, para in self.MTLS.named_parameters():
            total_params += para.numel()
            if "model_encoder" in name:
                model_encoder_params += [para]
                para.requires_grad = False
            elif "model_embedding" in name:
                model_embedding_params += [para]
                para.requires_grad = False
            elif "SSS_embedding" in name:
                SSS_embedding_params += [para]
                para.requires_grad = True
            else:
                raise ValueError("Unknown parameter name: {}".format(name))

            # If the parameter requires gradients
            if para.requires_grad:
                trainable_params += para.numel()

        # Print the total, trainable, and percentage of trainable parameters
        print('total params = {:e}, trainable params = {:e}, Percentage of trainable parameters = {:.4f}%'
              .format(total_params, trainable_params, trainable_params / total_params))

        # Initialize the parameters for the SSS_embedding
        params = {"params": SSS_embedding_params, "lr": self.settings.learning_rate},

        # Initialize the optimizer
        self.optimizer = torch.optim.AdamW(params,
                                           betas=(self.settings.beta1, self.settings.beta2),
                                           eps=self.settings.epsilon,
                                           weight_decay=self.settings.l2)

        # Initialize the scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer,
                                           mode='min',
                                           factor=self.settings.factor,
                                           patience=self.settings.patience,
                                           min_lr=self.settings.min_lr)

    def _store_settings(self):
        with open(self.settings.output_dir + "settings.json", "w") as fh:
            json.dump({k: v for k, v in self.settings.__dict__.items() if k not in "device".split()}, fh)

    def _run_train_batch(self, batch, optimizer):
        optimizer.zero_grad()
        # label [batch x label x head_seq x dependent_seq]
        loss_dist, loss_spat = self.MTLS(batch)
        loss = loss_dist + loss_spat
        # backpropagate the loss
        loss.backward()
        loss = float(loss)
        optimizer.step()
        return loss, loss_dist, loss_spat

    def _run_train_epoch(self, data):
        self.MTLS.train()

        # Initialize the total loss, total loss distance, and total loss spatial to 0
        total_loss = 0
        total_loss_dist = 0
        total_loss_spat = 0

        # Iterate through the data
        for i, batch in enumerate(tqdm(data)):
            # Send the batch to the device
            batch.to(self.device)
            # Run the training batch and get the loss, loss distance, and loss spatial
            loss, loss_dist, loss_spat = self._run_train_batch(batch, self.optimizer)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            total_loss += loss
            total_loss_dist += loss_dist
            total_loss_spat += loss_spat

        return total_loss, total_loss_dist, total_loss_spat

    def save(self, path):
        state = {"model": self.MTLS.state_dict()}
        torch.save(state, self.settings.output_dir + path)

    def load(self, path, load_all=False):
        print("Restoring model from {}".format(path))
        state = torch.load(path)
        if not load_all:
            pretrained_model_dict = {}
            for name in state["model"]:
                # if "model_encoder" in name or "SSS_embedding" in name:
                if "SSS_embedding" in name:
                    pretrained_model_dict[name] = state["model"][name]
            self.MTLS.load_state_dict(pretrained_model_dict, strict=False)
        else:
            self.MTLS.load_state_dict(state["model"])

        self.MTLS = self.MTLS.to(self.settings.device)

    def pre_train(self):
        # Set the settings
        settings = self.settings
        # Print the training starting message
        print("Training is starting for {} steps using ".format(settings.step) +
              "{} with the following settings:".format(self.device))
        print()
        # Print the settings
        for key, val in settings.__dict__.items():
            print("{}: {}".format(key, val))
        print(flush=True)

        # Initialize the best loss and the best loss epoch
        best_loss = math.inf
        best_loss_epoch = 1
        early_stop = False

        # Calculate the number of batches in an epoch
        epoch_batch = len(self.train_loader)
        epochs = math.ceil(settings.step / epoch_batch)

        # Iterate through the epochs
        for epoch in range(1, epochs + 1):
            if not early_stop:
                print("#" * 50)
                print("Epoch:{}".format(epoch))
                # Run the training epoch
                total_loss, total_loss_dist, total_loss_spat = self._run_train_epoch(self.train_loader)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Print the total loss
                print("loss {:.4f}".format(total_loss))
                print("loss_dist:{:.4f}  loss_spat:{:.4f}".format(total_loss_dist, total_loss_spat))
                print('learning_rate: {}'.format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                ##################################################################
                # Change the learning rate
                self.scheduler.step(total_loss)
                ###########################################################
                # Check if the total loss is less than the best loss
                improvement = total_loss < best_loss
                # Calculate the elapsed epochs
                elapsed = epoch - best_loss_epoch
                # If the total loss is not less than the best loss
                if not improvement:
                    print("Have not seen any improvement for {} epochs".format(elapsed))
                    print("Best loss was {:.4f} seen at epoch #{}".format(best_loss, best_loss_epoch))
                    # If the elapsed epochs is equal to 20
                    if elapsed == 20:
                        early_stop = True
                        print("!!!!!!!!!!!!!!!!!!!!early_stop!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                else:
                    # Set the best loss and the best loss epoch
                    best_loss = total_loss
                    best_loss_epoch = epoch
                    # Print a message indicating that the best model has been saved
                    print("Saving {} model".format(best_loss_epoch))
                    self.save("pretrain_best_model.save")
                    # Print the best f1 score and the best loss epoch
                    print("Best f1 was {:.4f} seen at epoch #{}".format(best_loss, best_loss_epoch))
        self.save("pretrain_last_epoch.save")
