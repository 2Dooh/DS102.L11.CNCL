from base.agent import Agent
from graphs.models import *
from data_loader import *

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch

from torchvision.ops import nms


import os
import datetime

import pickle5 as pickle

class SlidingWindow(Agent):
    def __init__(self,
                 model,
                 model_args,
                 data_loader,
                 data_loader_args,
                 criterion_args,
                 criterion,
                 optimizer=None,
                 optimizer_args=None,
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
        self.model_info = model_args
        self.mode = mode
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.report_freq = report_freq
        self.validate_every = validate_every
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.empty_cache = empty_cache
        self.save_threshold = save_threshold
        self.save_path = save_path

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # initialize counter
        self.current_epoch = self.current_iter = 1
        
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
        self.train_queue = data_loader.train_loader
        self.valid_queue = data_loader.test_loader

        # summary writer
        self.summary_writer = SummaryWriter(self.save_path) if summary_writer else None

        # default messages
        self.validate_msg = '{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'
        self.train_msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'

        callback(self) if callback else ...

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
        self.save_threshold = checkpoint['save_threshold']
            

    def save_checkpoint(self, error_rate, best_err):
        checkpoint = {"epoch": self.current_epoch,
                      "model_state_dict": self.model.state_dict(),
                      "optimizer_state_dict": self.optimizer.state_dict(),
                      "loss": self.criterion,
                      "save_threshold": best_err}

        name = '{}-Ep_{:03d}-Err_{:.3f}.pth.tar'
        filepath = os.path.join(self.save_path, name.format(self.model.__class__.__name__,
                                                            self.current_epoch,
                                                            error_rate))
        torch.save(checkpoint, filepath)
        
    def run(self):
        try:
            self.train() if self.mode == 'train' else self.validate()
        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize") 

    def train(self):
        best_err = valid_err = self.save_threshold
        while self.current_epoch <= self.max_epochs:
            self.train_one_epoch()
            if self.scheduler:
                self.scheduler.step()
                print('Epoch: {} - lr {}'.format(self.current_epoch, 
                                                 self.scheduler.get_last_lr()[0]))
            if self.current_epoch % self.validate_every == 0:
                valid_err, _ = self.validate()
            
            if valid_err < best_err:
                best_err = valid_err
                self.save_checkpoint(valid_err, best_err)

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
                print(self.train_msg.format(self.current_epoch, 
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

        self.current_epoch = 1

        with torch.no_grad():
            for step, inputs in enumerate(self.valid_queue):

                # inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.to(self.device)
                # print(inputs.size())
                outputs = self.model(inputs)
                # print(outputs.size())
                # test_loss += self.criterion(outputs, targets.view_as(outputs[:, 1])).item()  # sum up batch loss

                predicted = self.predict(outputs)
                print(predicted)
                # total += targets.size(0)
                # correct += predicted.eq(targets.view_as(predicted)).sum().item()

            avg_loss = test_loss/total
            err = 100.*(1- (correct/total))
            if self.verbose:
                acc = 100.*correct/total
                print(self.validate_msg.format('Test', avg_loss, correct, total, acc))

        
        if self.summary_writer:
            self.summary_writer.add_scalar('Loss/test', avg_loss, self.current_epoch)
            self.summary_writer.add_scalar('Error_rate/test', err, self.current_epoch)

        return err, avg_loss
    
    def predict(self, outputs):
        outputs = torch.nn.Softmax2d()(outputs)
        scores, grid_pred = outputs.max(1)

        x, y = torch.where(grid_pred[0] != 3)
        max_x, max_y = self.data_info['input_size'][1:]
        norm_x = x / max_x
        norm_y = y / max_y
        start_point = torch.vstack((norm_x, norm_y)).T.cpu()

        img_size = torch.Tensor(self.data_info['input_size'][1:])
        window_size = torch.Tensor(self.model_info['input_size'][1:])
        end_point = start_point + (window_size / img_size) 
        boxes = torch.cat([start_point, end_point], dim=1)

        keep = nms(boxes=boxes.cuda(), 
                   scores=scores[0, x, y].flatten(), 
                   iou_threshold=.5)
        
        info = {'boxes': start_point[keep].cpu().numpy(), 
                'labels': grid_pred[0, x[keep], y[keep]].cpu().numpy(),
                'scores': scores[0, x[keep], y[keep]].cpu().numpy()}
        with open('./logs_fhd/pred_{}.pickle'.format(self.current_epoch), 'wb') as f:
          pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)
        res  = torch.unique(grid_pred, return_counts=True)
        self.current_epoch += 1
        return res

    def finalize(self):
        now = datetime.datetime.now()
        torch.save(self.model.state_dict(), os.path.join(self.save_path, '{}.pth.tar'.format(now.strftime("%Y%m%d-%H%M"))))