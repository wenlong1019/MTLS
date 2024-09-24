import math

import torch
from seqeval.metrics import f1_score

from src.model_interactor import ModelInteractor


class NerInteractor(ModelInteractor):
    def __init__(self, settings):
        super().__init__(settings)

    def train(self):
        settings = self.settings

        # Initialize the best F1 score and the epoch with the best F1 score
        best_f1 = 0
        best_f1_epoch = 1

        # Set the early stopping flag to False
        early_stop = False

        # Calculate the number of batches in an epoch and the number of epochs
        epoch_batch = len(self.train_loader)
        epochs = math.ceil(settings.step / epoch_batch)

        for epoch in range(1, epochs + 1):
            if not early_stop:
                print("#" * 50)
                print("Epoch:{}".format(epoch))

                # Run the training process for an epoch
                total_loss = self._run_train_epoch(self.train_loader)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print("loss {:.4f}".format(total_loss))
                print('LR_embedding {}'.format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                print('LR_encoder {}'.format(self.optimizer.state_dict()['param_groups'][1]['lr']))
                print('LR_other {}'.format(self.optimizer.state_dict()['param_groups'][2]['lr']))

                # change learning rate
                self.scheduler.step(total_loss)

                print("-----validation phase-----")
                predicted = self.predict(self.val_loader)
                preds_list, out_label_list = self.align_predictions(self.val_data, predicted)
                f1 = float(f1_score(out_label_list, preds_list))
                print("Dev f1 on epoch {} is {:.2%}".format(epoch, f1))

                improvement = f1 > best_f1
                elapsed = epoch - best_f1_epoch

                if not improvement:
                    print("Have not seen any improvement for {} epochs".format(elapsed))
                    print("Best f1 was {:.2%} seen at epoch #{}".format(best_f1, best_f1_epoch))
                    if elapsed == 20:
                        early_stop = True
                        print("!!!!!!!!!!!!!!!!!!!!early_stop!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                else:
                    best_f1 = f1
                    best_f1_epoch = epoch
                    print("Saving {} model".format(best_f1_epoch))
                    self.save("best_model.save")
                    print("Best f1 was {:.2%} seen at epoch #{}".format(best_f1, best_f1_epoch))
        self.save("last_epoch.save")

    def predict_val(self):
        predicted = self.predict(self.val_loader)
        preds_list, out_label_list = self.align_predictions(self.val_data, predicted)
        f1 = float(f1_score(out_label_list, preds_list))
        print("val f1 is {:.2%}".format(f1))

    def predict_test(self):
        predicted = self.predict(self.test_loader)
        preds_list, out_label_list = self.align_predictions(self.test_data, predicted)
        f1 = float(f1_score(out_label_list, preds_list))
        print("test f1 is {:.2%}".format(f1))

    def align_predictions(self, entries, predicted):
        label_map = self.settings.target_label_switch
        # Initialize lists to store predictions and labels
        preds_list = []
        out_label_list = []
        # Iterate through each entry in the entries list
        for entry in entries:
            pred = predicted[entry[0]].numpy()
            label = entry[1].numpy()
            preds = [label_map[p] for p in pred]
            out_labels = [label_map[i] for i in label]
            preds_list.append(preds)
            out_label_list.append(out_labels)
        return preds_list, out_label_list
