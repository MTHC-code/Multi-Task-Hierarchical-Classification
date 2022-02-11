import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Predict(nn.Module):

    def __init__(self, embed_dim=32, hidden_dim=64):

        super(Predict, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.disk_layer = nn.Sequential(
                            nn.Linear(embed_dim, self.hidden_dim),
                            nn.ReLU(),
                            nn.LSTM(self.hidden_dim, hidden_size=self.hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        )


        self.predict_head1 =  nn.Linear(self.hidden_dim * 2, 2)
        self.predict_head2 =  nn.Linear(self.hidden_dim * 2, 2)

                   

    def forward(self, node_input, is_train = True, tp_weight = 0.7):
        node_size = 10
        feature_dim = 32
        
        tot_disk_feature = []
        for i in range(node_size):
            cur_disk = node_input[:,:,feature_dim*i:feature_dim*(i+1)]
            disk_feature, (h, c) = self.disk_layer(cur_disk)
            disk_feature = disk_feature.mean(dim=1)
            tot_disk_feature.append(disk_feature)
            
       

        tot_pred_1, tot_pred_2 = [], []

        for i, disk_feature in enumerate(tot_disk_feature):

            pred_1 = self.predict_head1(disk_feature)  
            pred_2 = self.predict_head2(disk_feature)  
           

            tot_pred_1.append(pred_1)
            tot_pred_2.append(pred_2)

        tot_pred_1 = torch.stack(tot_pred_1, dim=0).permute(1,0,2)
        tot_pred_2 = torch.stack(tot_pred_2, dim=0).permute(1,0,2)
        return  tot_pred_1, tot_pred_2


def train_model(train_loader, model, epoch, criterion, optimizer, device):
    model.train()
    running_loss_1 = 0.0
    running_loss_2 = 0.0
    running_loss_node = 0.0

    for batch_idx, (node_data, label_1, label_2, label_node) in enumerate(train_loader):

        label_1 = label_1.to(device).to(torch.long)
        label_2 = label_2.to(device).to(torch.long)
        label_node = label_node.to(device).to(torch.long)
       
        tot_label = [label_1, label_2]
        
        out_1, out_2= model(node_data)
        tot_output = [out_1, out_2]


        loss_1 = criterion(out_1.reshape(-1,2), label_1.reshape(-1,))
        loss_2 = criterion(out_2.reshape(-1,2), label_2.reshape(-1,))
        
      
        loss = loss_1 + loss_2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss_1 += loss_1 * label_1.size(0) 
        running_loss_2 += loss_2 * label_2.size(0) 
        if batch_idx % 20 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.4f}, Loss2: {:.4f}".format(
                epoch, batch_idx * label_1.size(0), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), running_loss_1/640,running_loss_2/640
            ))
            running_loss_1 = 0.0
            running_loss_2 = 0.0
            running_loss_node = 0.0

def valid_model(valid_loader, model, epoch, criterion, device, tp_weight=0.2):
    model.eval()
    running_loss_1 = 0.0
    running_loss_2 = 0.0

    tot_score_res = [ [],[],[] ]
    tot_score_device = []
    tot_labels = [ [],[],[] ]
    tot_label_node = []
    tot_top_acc = []
    with torch.no_grad():
        for batch_idx, (node_data, label_1, label_2, label_node) in enumerate(valid_loader):
            label_1 = label_1.to(device).to(torch.long)
            label_2 = label_2.to(device).to(torch.long)
            label_node = label_node.to(device).to(torch.long)
            out_1, out_2  = model(node_data, is_train=False, tp_weight=tp_weight)
           
            tot_output = [out_1, out_2]
            tot_label = [label_1, label_2]

            loss_1 = criterion(out_1.reshape(-1,2), label_1.reshape(-1,))
            loss_2 = criterion(out_2.reshape(-1,2), label_2.reshape(-1,))

            running_loss_1 += loss_1 * label_1.size(0) 
            running_loss_2 += loss_2 * label_2.size(0) 

            
            i = 0
            for out, label in zip(tot_output, tot_label):
                label = label.reshape(-1,).detach().cpu().numpy()
                prob = F.softmax(out.reshape(-1,2), dim=1).detach().cpu().numpy()[:, 1]
                if batch_idx == 0:
                    tot_labels[i] = label
                    tot_score_res[i] = prob
                else:
                    tot_labels[i] = np.concatenate((tot_labels[i], label), axis=0)
                    tot_score_res[i] = np.concatenate((tot_score_res[i], prob), axis=0)
                i += 1

        np.save('multi_result/epoch_{}_multi_predict_result.npy'.format(epoch),np.array(tot_score_res))
        np.save('multi_result/epoch_{}_multi_label.npy'.format(epoch),np.array(tot_labels))
      
        epoch_loss = (running_loss_1 + running_loss_2) / len(valid_loader.dataset)
        epoch_loss_1 = running_loss_1 / len(valid_loader.dataset)
        epoch_loss_2 = running_loss_2 / len(valid_loader.dataset)

        print("----------------------------")
        print("Valid Epoch: {} Loss: {:.4f} Loss_1: {:.4f} Loss_2: {:.4f}".format(
            epoch, epoch_loss, epoch_loss_1, epoch_loss_2))
        


        task = 1
        prf_list = []
        for label, score in zip(tot_labels, tot_score_res):
            best_f1,best_pre,best_rec = 0,0,0
            for threhhold in range(1000):
                threshold = 0.001*threhhold
                pred = (score > threshold).astype(int)
                TP = ((pred == 1).astype(int) + (label == 1).astype(int) == 2).sum()
                FP = ((pred == 1).astype(int) + (label == 0).astype(int) == 2).sum()
                FN = ((pred == 0).astype(int) + (label == 1).astype(int) == 2).sum()
                
                precision = 1.0 * TP / (TP + FP)
                recall = 1.0 * TP / (TP + FN)
                if precision==0 or recall==0:continue
                F1 = 2.0 * precision * recall / (precision + recall)
                if F1 > best_f1:
                    best_f1 = F1
                    best_pre = precision
                    best_rec = recall
            prf_list.append([best_f1, best_pre, best_rec])
            sort_id = score.argsort()[::-1]
            sort_label = label[sort_id]
            top_k = label.sum()
            top_acc = 1.0 * sort_label[:top_k].sum() / top_k
            tot_top_acc.append(top_acc)
            print("task_{} ------> top_acc: {:.2f}%, F1: {:.2f}%, Precision: {:.2f}% Recall: {:.2f}%,TP:{},FP:{},FN:{}".format(
                task, top_acc*100, best_f1*100, best_pre*100, best_rec*100, TP,FP,FN
            ))
            task += 1
        print("----------------------------")

    return tot_top_acc, prf_list

