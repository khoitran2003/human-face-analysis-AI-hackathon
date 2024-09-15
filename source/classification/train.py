import os
import numpy as np
import torch
import torch.nn as nn
from src.classification.resnet_50_modify import Modified_Resnet_50
from src.classification.focal_loss import focal_loss
from tqdm.autonotebook import tqdm
from src.classification.data_loader_clf import Get_Loader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import shutil
import warnings
warnings.simplefilter('ignore')

class Classify():
    def __init__(self, cfg: dict, label_cfg: dict, device):

        self.dataloader = Get_Loader(cfg, label_cfg)
        self.num_epochs = cfg['train']['num_epochs']
        self.learning_rate = cfg['train']['learning_rate']
        self.device = device

        self.model = Modified_Resnet_50().to(self.device)
        self.optim = torch.optim.SGD(self.model.parameters(), self.learning_rate)
        self.tensorboad_path = cfg['train']['log_path']
        self.checkpoint_path = cfg['train']['checkpoint']
        self.cfg = cfg
        self.label_cfg = label_cfg

    def train(self):
        train_loader, val_loader = self.dataloader.load_train_val()
        optimizer = self.optim

        if os.path.isdir(os.path.join(self.checkpoint_path, 'classifier_last.pt')):
            checkpoint = torch.load(os.path.join(self.checkpoint_path, 'classifier_last.pt'))
            start_epoch = checkpoint['epochs']
            self.model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_loss = checkpoint['best_loss']
        else:
            start_epoch = 0
            best_loss = 1000

        if os.path.isdir(self.tensorboad_path):
            shutil.rmtree(self.tensorboad_path)
        else:
            os.makedirs(self.tensorboad_path)

        writer = SummaryWriter(self.tensorboad_path)

        for epoch in range (start_epoch, self.num_epochs):
            self.model.train()
            train_progress_bar = tqdm(train_loader, colour='cyan')
            for iter, batch in enumerate(train_progress_bar):
                images, labels = batch
                images, labels = images.to(self.device), labels.to(device=self.device, dtype=torch.long)
                age_out, gender_out, emotion_out = self.model(images)
                age_loss = focal_loss('age', self.cfg, age_out, labels[:, 0]).to(self.device)
                gender_loss = focal_loss('gender', self.cfg, gender_out, labels[:, 1]).to(self.device)
                emotion_loss = focal_loss('emotion', self.cfg, emotion_out, labels[:, 2]).to(self.device)
                total_loss = (age_loss + gender_loss + emotion_loss)/3

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                train_progress_bar.set_description('Epoch: {}/{}. Total Loss: {}'.format(epoch+1, self.num_epochs, total_loss.item()))
                writer.add_scalar('Train/Loss', total_loss.item(), iter + epoch * len(train_loader))
            
            self.model.eval()
            with torch.no_grad():
                val_progress_bar = tqdm(val_loader, colour='yellow')
                all_loss = []
                age_accs = []
                gender_accs = []
                emotion_accs = []
                for iter, batch in enumerate(val_progress_bar):
                    images, labels = batch
                    images, labels = images.to(self.device), labels.to(device=self.device, dtype=torch.long)
                    age_out, gender_out, emotion_out = self.model(images)
                    age_loss = focal_loss('age', self.cfg, age_out, labels[:, 0]).to(self.device)
                    gender_loss = focal_loss('gender', self.cfg, gender_out, labels[:, 1]).to(self.device)
                    emotion_loss = focal_loss('emotion', self.cfg, emotion_out, labels[:, 2]).to(self.device)
                    total_loss = (age_loss + gender_loss + emotion_loss)/3
                    
                    age_pred = torch.argmax(age_out, dim=-1)
                    gender_pred = torch.argmax(gender_out, dim=-1)
                    emotion_pred = torch.argmax(emotion_out, dim=-1)

                    age_acc = accuracy_score(labels[:, 0], age_pred)
                    gender_acc = accuracy_score(labels[:, 1], gender_pred)
                    emotion_acc = accuracy_score(labels[:, 2], emotion_pred)

                    all_loss.append(total_loss.item())
                    age_accs.append(age_acc)
                    gender_accs.append(gender_acc)
                    emotion_accs.append(emotion_acc)

                
                all_loss = np.mean(all_loss)
                age_accs = np.mean(age_accs)
                gender_accs = np.mean(gender_accs)
                emotion_accs = np.mean(emotion_accs)
                print('Epoch: {}. Total Loss: {}'.format(epoch+1, total_loss.item()))
                writer.add_scalar('Val/Loss', total_loss.item(), epoch+1)
                writer.add_scalar('Age_Acc/Loss', age_accs, epoch+1)
                writer.add_scalar('Gender_Acc/Loss', gender_accs, epoch+1)
                writer.add_scalar('Emotion_Acc/Loss', emotion_accs, epoch+1)

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss
            }

            torch.save(checkpoint, os.path.join(self.checkpoint_path, 'classifier_last.pt'))

            if all_loss < best_loss:
                torch.save(checkpoint, os.path.join(self.checkpoint_path, 'classifier_best.pt'))
                best_loss = all_loss
        
        print('Training finish...')
        


                




        