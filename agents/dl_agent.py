from base.agent import Agent
from graphs.models import *
from data_loader import *

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
from torch import optim
from tensorboardX import SummaryWriter
import torch
import torchvision

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os
import datetime

import pickle5 as pickle

from functools import reduce

from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize

import warnings
warnings.filterwarnings("ignore")

class DeepLearningAgent(Agent):
    def __init__(self,
                 model,
                 model_args,
                 data_loader,
                 data_loader_args,
                 optimizer,
                 optimizer_args,
                 criterion_args,
                 criterion,
                 mode='train',
                 seed=1,
                 cuda=False,
                 save_agent=True,
                 save_threshold=10,
                 empty_cache=False,
                 max_epochs=1,
                 validate_every=1,
                 verbose=False,
                 deterministic=False, 
                 scheduler=None,
                 scheduler_args={},
                 grad_clip=None,
                 report_freq=1, 
                 summary_writer=False,
                 checkpoint_file=None,
                 save_path='./pretrained_weights',
                 callback=None,
                 **kwargs):

        # set cuda flag
        self.has_cuda = torch.cuda.is_available()
        if self.has_cuda and not cuda:
            print("WARNING: You have a CUDA device, so you should enable CUDA")

        self.cuda = self.has_cuda and cuda

        # set manual seed
        self.manual_seed = seed
        torch.manual_seed(self.manual_seed)
        if self.cuda:
            cudnn.enabled = True
            cudnn.benchmark = not deterministic
            cudnn.deterministic = deterministic
            if deterministic:
              print('applying deterministic mode; cudnn disabled!')
        
        # save important parameter
        self.data_info = data_loader_args
        self.mode = mode
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.report_freq = report_freq
        self.validate_every = validate_every
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.empty_cache = empty_cache
        self.save_threshold = save_threshold
        self.scores_table = [[20] * 5]
        self.save_path = save_path

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # initialize counter
        self.current_epoch = self.current_iter = 0
        
        self.criterion = getattr(nn, criterion, None)(**criterion_args)
        self.model = globals()[model](**model_args)
        self.parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = getattr(optim, optimizer, None)(self.parameters,
                                                         **optimizer_args)
        # get device
        self.device = torch.device("cuda:0" if self.cuda else "cpu")
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        print("Program will run on *****{}*****".format(self.device))

        # load checkpoint
        self.load_checkpoint(checkpoint_file)

        if scheduler:
            self.scheduler = getattr(lr_scheduler, scheduler)(optimizer=self.optimizer, 
                                                              **scheduler_args)

        # save agent
        self.save_agent() if save_agent else ...
                
        data_loader = globals()[data_loader](**data_loader_args)
        self.class_labels = data_loader.class_labels
        self.train_queue = data_loader.train_loader
        self.valid_queue = data_loader.test_loader

        # summary writer
        self.summary_writer = SummaryWriter(self.save_path) if summary_writer else None
        if summary_writer:
            #   dummy_input = torch.randn(1, *self.data_info['input_size'], device='cpu')
            #   assert isinstance(self.model.cpu(), torch.nn.Module)
            #   self.summary_writer.add_graph(self.model.to('cpu'), (dummy_input, ))

            data_iter = iter(self.train_queue)
            images, labels = data_iter.next()
            img_grid = torchvision.utils.make_grid(images)

            # show images
            # self.matplotlib_imshow(img_grid, one_channel=True)

            # write to tensorboard
            self.summary_writer.add_image('images', img_grid)
            class_labels = [self.class_labels[lab] for lab in labels]
            features = images.view(-1, reduce(lambda x, y: x * y, self.data_info['input_size']))
            self.summary_writer.add_embedding(features,
                                            metadata=class_labels,
                                            label_img=images)

            
            self.summary_writer.close()
            # pass

        # default messages
        self.validate_msg = '{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'
        self.train_msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'

        callback(self) if callback else ...


    @staticmethod
    def matplotlib_imshow(img, one_channel=False):
        if one_channel:
            img = img.mean(dim=0)
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def save_agent(self):
        now = datetime.datetime.now()
        name = '{}_{}.pickle'.format(self.__class__.__name__, now.strftime("%Y%m%d-%H%M"))
        with open(os.path.join(self.save_path, name), 'wb') as handle:
            pickle.dump(self, handle, pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, path=None):
        if not path:
          return None
        checkpoint = torch.load(path, map_location=self.device, pickle_module=pickle)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.criterion = checkpoint['loss']
        # self.save_threshold = checkpoint['save_threshold']
        self.scores_table = checkpoint['scores_table']
            

    def save_checkpoint(self, error_rate, scores_table):
        checkpoint = {"epoch": self.current_epoch-1,
                      "model_state_dict": self.model.state_dict(),
                      "optimizer_state_dict": self.optimizer.state_dict(),
                      "loss": self.criterion,
                      "scores_table": scores_table,
                      "error_scores": error_rate}

        name = '{}-Ep_{:03d}-Err_{:.3f}.pth.tar'
        filepath = os.path.join(self.save_path, name.format(self.model.__class__.__name__,
                                                            self.current_epoch-1,
                                                            error_rate.mean()))
        torch.save(checkpoint, filepath, pickle_module=pickle, 
                   pickle_protocol=pickle.HIGHEST_PROTOCOL,
                   _use_new_zipfile_serialization=True)
        
    def run(self):
        try:
            self.train() if self.mode == 'train' else self.validate()
        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize") 

    def train(self):
        # best_err = valid_err = [10] * 5
        # self.scores_table = [[20] * 5]
        while self.current_epoch < self.max_epochs:
            self.train_one_epoch()
            if self.scheduler:
                self.scheduler.step()
                print('Epoch: {} - lr {}'.format(self.current_epoch, 
                                                 self.scheduler.get_last_lr()[0]))
            if self.current_epoch % self.validate_every == 0:
                valid_err, _ = self.validate()
                with open(os.path.join(self.save_path, 'scores_table.txt'), 'a+') as handle:
                    handle.write(','.join(str(metric) for metric in valid_err) + '\n')
            self.scores_table += [valid_err]
            save = True
            for score in self.scores_table:
                if self.dominate(np.array(score), np.array(valid_err)):
                    save = False
                    break
            if save:
                self.save_checkpoint(valid_err, self.scores_table)

        if self.empty_cache:
          torch.cuda.empty_cache()


    def train_one_epoch(self):
        self.model.train()
        correct = total = train_loss = 0
        n_inputs = len(self.train_queue.dataset)
        for step, (inputs, targets) in enumerate(self.train_queue):
            targets, loss, outputs = self.feed_forward(inputs, targets)

            train_loss += loss.item()
            predicted = self.predict(outputs)
            total += targets.size(0)
            correct += predicted.eq(targets.view_as(predicted)).sum().item()
            
            if self.verbose and step % self.report_freq == 0:
                percentage = 100.*total/n_inputs
                print(self.train_msg.format(self.current_epoch+1, 
                                            total, 
                                            n_inputs, 
                                            percentage, 
                                            loss.item()))

            self.current_iter += 1
        
        avg_loss = train_loss/total
        err = 100.*(1- (correct/total))
        if self.verbose:
            acc = 100.*correct/total
            print(self.validate_msg.format('Train', avg_loss, correct, total, acc))
        if self.summary_writer:
            self.summary_writer.add_scalar('Loss/train', avg_loss, self.current_epoch)
            self.summary_writer.add_scalar('Error_rate/train', err, self.current_epoch)

        self.current_epoch += 1
        return err, avg_loss

    def feed_forward(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets.view_as(outputs[:, 1]))

        loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), 
                                     self.grad_clip)
        self.optimizer.step()

        return targets, loss, outputs

    def validate(self):
        self.model.eval()
        test_loss = correct = total = 0

        y_true, y_score = [], []
        with torch.no_grad():
            for step, (inputs, targets) in enumerate(self.valid_queue):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                test_loss += self.criterion(outputs, targets.view_as(outputs[:, 1])).item()  # sum up batch loss

                predicted = self.predict(outputs)
                total += targets.size(0)
                correct += predicted.eq(targets.view_as(predicted)).sum().item()

                y_true += [targets.cpu().numpy()]
                y_score += [predicted.view_as(targets).cpu().numpy()]
                print(total)

            y_true = np.concatenate(y_true)
            y_score = np.concatenate(y_score)
            onehot_targets = label_binarize(y_true, classes=range(y_true.max(axis=0)+1))
            onehot_pred = label_binarize(y_score, classes=range(y_true.max(axis=0)+1))
            
            precision = precision_score(y_true, y_score, average=None)
            print(precision)
            recall = recall_score(y_true, y_score, average=None)
            print(recall)

            ap = average_precision_score(y_true=onehot_targets, 
                                          y_score=onehot_pred, 
                                          average='weighted')
            f1 = f1_score(y_true=y_true,
                          y_pred=y_score,
                          average='weighted')

            roc_auc_ovo = roc_auc_score(y_true=y_true,
                                        y_score=onehot_pred,
                                        average='weighted',
                                        multi_class='ovo')
            roc_auc_ovr = roc_auc_score(y_true=y_true,
                                        y_score=onehot_pred,
                                        average='weighted',
                                        multi_class='ovr')
            # mAP = ap / (y_true.max(axis=0) + 1)
                # correct = roc_auc_score(y_true=targets.flatten().cpu().numpy(),
                #                         y_score=onehot_pred, 
                #                         average='weighted', 
                #                         multi_class='ovo')
                # correct += predicted.eq(targets.view_as(predicted)).sum().item()
            acc = correct / total
            score = np.array([acc, ap, f1, roc_auc_ovo, roc_auc_ovr])
            avg_loss = test_loss/total
            err = 100.*(1 - score)
            if self.verbose:
                print(self.validate_msg.format('Test', avg_loss, correct, total, acc*100))
                print('Acc: {:.3f}% - AP: {:.3f}% - F1: {:.3f}% - ROC_AUC(OVR/OVO): {:.3f}%/{:.3f}%\n'.format(acc*100,
                                                                             ap*100, 
                                                                             f1*100,
                                                                             roc_auc_ovr*100,
                                                                             roc_auc_ovo*100))

        
        if self.summary_writer:
            self.summary_writer.add_scalar('Loss/test', avg_loss, self.current_epoch)
            self.summary_writer.add_scalar('Error_rate/test', 100.*(1 - (correct/total)), self.current_epoch)
            self.summary_writer.add_scalar('AP/Weighted/test', 100.*ap, self.current_epoch)
            self.summary_writer.add_scalar('F1/Weighted/test', 100.*f1, self.current_epoch)
            self.summary_writer.add_scalar('ROC_AUC_OVR/Weighted/test', 100.*roc_auc_ovr, self.current_epoch)
            self.summary_writer.add_scalar('ROC_AUC_OVO/Weigted/test', 100.*roc_auc_ovo, self.current_epoch)

        return err, avg_loss

    @staticmethod
    def dominate(score1, score2):
        not_dominated = score1 <= score2
        dominate = score1 < score2
        return not_dominated.all() and True in dominate
    
    def predict(self, outputs):
        _, pred = outputs.max(1)
        return pred

    def finalize(self):
        now = datetime.datetime.now()
        torch.save(self.model.state_dict(), os.path.join(self.save_path, '{}.pth.tar'.format(now.strftime("%Y%m%d-%H%M"))))