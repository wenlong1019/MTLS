import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import Datasets
from src.loss_functions import cross_entropy_loss
from src.mtls import MTLS
from src.utils.padded_collate import padded_collate


class ModelInteractor:
    def __init__(self, settings):
        # Initialize the train_data, val_data, and test_data attributes
        self.train_data = None
        self.val_data = None
        self.test_data = None

        # Initialize the train_loader, val_loader, and test_loader attributes
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Initialize the optimizer attribute
        self.optimizer = None

        # Initialize the scheduler attribute
        self.scheduler = None

        # Store the settings object
        self.settings = settings
        self.device = settings.device
        self.batch_size = settings.batch_size
        self.step = settings.step

        # Initialize the MTLS object and store it
        self.MTLS = MTLS(settings)
        self.MTLS = self.MTLS.to(self.settings.device)

        # Initialize the data, optimizer, and scheduler
        self._init_data(settings)
        self._init_optimizer()
        self._store_settings()

    def _init_data(self, settings):
        # If the train_file is not None, initialize the train_data and train_loader attributes
        if settings.train_file is not None:
            self.train_data = Datasets(settings.train_file, mode="train", settings=self.settings)
            self.train_loader = DataLoader(self.train_data,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           collate_fn=padded_collate)
        # If the val_file is not None, initialize the val_data and val_loader attributes
        if settings.val_file is not None:
            self.val_data = Datasets(settings.val_file, mode="dev", settings=self.settings)
            self.val_loader = DataLoader(self.val_data,
                                         batch_size=self.batch_size,
                                         shuffle=False,
                                         collate_fn=padded_collate)
        # If the test_file is not None, initialize the test_data and test_loader attributes
        if settings.test_file is not None:
            self.test_data = Datasets(settings.test_file, mode="test", settings=self.settings)
            self.test_loader = DataLoader(self.test_data,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          collate_fn=padded_collate)

    def _init_optimizer(self):
        # Initialize lists to store the parameters of the encoder, embedding, SSS_embedding, and other layers
        model_encoder_params = []
        model_embedding_params = []
        SSS_embedding_params = []
        other_params = []

        # Initialize the total and trainable parameter counts
        total_params = 0
        trainable_params = 0

        # Iterate through the parameters of the model
        for name, para in self.MTLS.named_parameters():
            total_params += para.numel()
            if "model_encoder" in name:
                model_encoder_params += [para]
                para.requires_grad = True
            elif "model_embedding" in name:
                model_embedding_params += [para]
                para.requires_grad = False
            elif "SSS_embedding" in name:
                SSS_embedding_params += [para]
                if "position_embeddings" in name:
                    para.requires_grad = False
                else:
                    para.requires_grad = True
            else:
                other_params += [para]
                para.requires_grad = True

            # If the parameter requires grad, add it to the trainable parameter count
            if para.requires_grad:
                trainable_params += para.numel()

        # Print the total, trainable, and percentage of trainable parameters
        print('total params = {:e}, trainable params = {:e}, Percentage of trainable parameters = {:.4f}%'
              .format(total_params, trainable_params, trainable_params / total_params))

        # Initialize the parameter list for the optimizer
        params = [{"params": SSS_embedding_params, "lr": self.settings.lr_embeddings},
                  {"params": model_encoder_params, "lr": self.settings.lr_encoder},
                  {"params": other_params, "lr": self.settings.lr_other}]

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
        label_scores = self.MTLS(batch)

        # Calculate loss
        loss = cross_entropy_loss(label_scores, batch)
        loss.backward()
        loss = float(loss)
        optimizer.step()
        return loss

    def _run_test_batch(self, batch):
        label_scores = self.MTLS(batch)
        predictions = {}
        # Iterate through each sample in the batch
        for i, size in enumerate(batch.seq_lengths):
            size = size.item()
            scores = torch.nn.functional.softmax(label_scores[i, 1:size, :], dim=1).cpu()
            prediction = torch.argmax(scores, dim=1).float()
            predictions[batch.ids[i]] = prediction
        return predictions

    def _run_train_epoch(self, data):
        self.MTLS.train()

        # Initialize the total loss
        total_loss = 0

        # Iterate through the data
        for i, batch in enumerate(tqdm(data)):
            batch.to(self.device)

            loss = self._run_train_batch(batch, self.optimizer)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            total_loss += loss

        return total_loss

    def predict(self, data_loader):
        self.MTLS.eval()
        predictions = {}
        # Iterate through the data loader
        for batch in data_loader:
            batch.to(self.device)
            with torch.no_grad():
                pred = self._run_test_batch(batch)
                predictions.update(pred)
        return predictions

    def save(self, path):
        state = {"model": self.MTLS.state_dict()}
        torch.save(state, self.settings.output_dir + path)

    def load(self, path, load_all=False):
        print("Restoring model from {}".format(path))
        # load the model state from the given path
        state = torch.load(path)
        # if the load all flag is not set, create an empty dictionary
        if not load_all:
            pretrained_model_dict = {}
            for name in state["model"]:
                # if the key contains "SSS_embedding", add it to the dictionary
                if "SSS_embedding" in name and "position_embeddings" not in name:
                    pretrained_model_dict[name] = state["model"][name]
            # load the dictionary into the model
            self.MTLS.load_state_dict(pretrained_model_dict, strict=False)
        else:
            self.MTLS.load_state_dict(state["model"])

        self.MTLS = self.MTLS.to(self.settings.device)
