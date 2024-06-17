import math

import torch

from src.model_interactor import ModelInteractor


def calculate_accuracy(entries, predicted):
    correct = 0
    all_item = 0
    for entry in entries:
        pred = predicted[entry[0]].numpy()
        label = entry[1].numpy()
        correct += (pred == label[:len(pred)]).sum().item()
        all_item += len(pred)
    return correct / all_item


class PosInteractor(ModelInteractor):
    def __init__(self, settings):
        super().__init__(settings)

    def train(self):
        stage = 2
        self._init_optimizer(stage)
        settings = self.settings
        ################################################################
        best_acc = 0
        best_acc_epoch = 1
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
                print('LR_textual_embedding {}'.format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                print('LR_textual_encoder {}'.format(self.optimizer.state_dict()['param_groups'][1]['lr']))
                print('LR_textual_other {}'.format(self.optimizer.state_dict()['param_groups'][2]['lr']))
                ##################################################################
                # change learning rate
                self.scheduler.step(total_loss)
                ###########################################################
                print("-----validation phase-----")
                predicted = self.predict(self.val_loader)
                acc = calculate_accuracy(self.val_data, predicted)
                print("Primary Dev acc on epoch {} is {:.2%}".format(epoch, acc))

                improvement = acc > best_acc
                elapsed = epoch - best_acc_epoch

                if not improvement:
                    print("Have not seen any improvement for {} epochs".format(elapsed))
                    print("Best acc was {:.2%} seen at epoch #{}".format(best_acc, best_acc_epoch))
                    if elapsed == 20:
                        early_stop = True
                        print("!!!!!!!!!!!!!!!!!!!!early_stop!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                else:
                    best_acc = acc
                    best_acc_epoch = epoch
                    print("Saving {} model".format(best_acc_epoch))
                    self.save("best_model.save")
                    print("Best acc was {:.2%} seen at epoch #{}".format(best_acc, best_acc_epoch))
        self.save("last_epoch.save")

    def predict_val(self):
        predicted = self.predict(self.val_loader)
        acc = calculate_accuracy(self.val_data, predicted)
        print("val acc is {:.2%}".format(acc))

    def predict_test(self):
        predicted = self.predict(self.test_loader)
        acc = calculate_accuracy(self.test_data, predicted)
        print("test acc is {:.2%}".format(acc))
