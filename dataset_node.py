import torch
import numpy as np
import pandas as pd
from copy import deepcopy
import os
import copy

class PreDataSet(torch.utils.data.Dataset):

    def __init__(self, data_path_list):
        SEED = 3047
        np.random.seed(SEED)
        self.data = []
        tot_node_feature = []
        tot_y1 = []
        tot_y2 = []
        tot_node = []
        if not os.path.exists('cache'):
            os.mkdir('cache')

      
        self.node_name_list = []
        for data_path in data_path_list:
            cur_tot_node_feature = []
            cur_tot_y1 = []
            cur_tot_y2 = []
            cur_tot_node = []

            data_path = os.path.join('node_data', data_path + '.csv')
            result = pd.read_csv(data_path)
            node_size = 10
            node_result = result.groupby(['env_cloud_roleInstance'])
            for _, result in node_result:

                for i in range(node_size):
                    self.node_name_list.append(result.iloc[0]['env_cloud_roleInstance'])

                d_y1 = result.iloc[:,0].values.astype(int)
                d_y2 = result.iloc[:,1].values.astype(int)
                
                d_x = result.iloc[:,8:].values 
                d_x[np.isnan(d_x)] = 0.0

                d_x[:,:1008] /= 200
                d_x[:,1080:1224] /= 200
                d_x[:,1512:1728] /= 200

                feature_dim = len(d_x[0]) // 72 
                d_feature = np.zeros((d_x.shape[0],72,feature_dim))
                for i in range(d_x.shape[0]):
                    d_feature[i] = np.transpose(d_x[i].reshape(feature_dim, 72))
                
                choose_idx = []
                node_feature = np.zeros((72,feature_dim*node_size))
                pos_disk = np.where(d_y1+d_y2>0)[0]
                for idx, val in enumerate(pos_disk):
                    if idx==node_size:break
                    try:
                        node_feature[:,feature_dim*idx:feature_dim*(idx+1)] = deepcopy(d_feature[val])
                        choose_idx.append(val)
                    except:
                        import pdb;pdb.set_trace()
                neg_disk = np.where(d_y1+d_y2==0)[0]
                remain_len = max(node_size - len(pos_disk),0)
                for idx, val in enumerate(neg_disk[:remain_len]):
                    node_feature[:,feature_dim*(idx+len(pos_disk)):feature_dim*(idx+len(pos_disk)+1)] = deepcopy(d_feature[val])
                    choose_idx.append(val)

                choose_idx = np.array(choose_idx)
                d_y1, d_y2 = d_y1[choose_idx], d_y2[choose_idx]
                node_class = (d_y1.sum()+d_y2.sum()>0)
                
                cur_tot_node_feature.append(node_feature)
                cur_tot_y1.append(d_y1)
                cur_tot_y2.append(d_y2)
                cur_tot_node.append(node_class)
               
            cur_tot_node_feature = np.stack(cur_tot_node_feature, axis=0)
            cur_tot_y1 = np.stack(cur_tot_y1, axis=0)
            cur_tot_y2 = np.stack(cur_tot_y2, axis=0)
            
            cur_tot_node = np.stack(cur_tot_node, axis=0).astype(int)
            print('data_path is {}'.format(data_path))
            print('node_feature shape is {}'.format(cur_tot_node_feature.shape)) 
            print('label1:pos num {}/{}'.format(cur_tot_y1.sum(),node_size*len(cur_tot_y1)))
            print('label2:pos num {}/{}'.format(cur_tot_y2.sum(),node_size*len(cur_tot_y2)))
            
            print('label_node:pos num {}/{}'.format(cur_tot_node.sum(),len(cur_tot_node)))

            if len(tot_node_feature)==0:
                tot_node_feature, tot_y1, tot_y2 tot_node = cur_tot_node_feature, cur_tot_y1, cur_tot_y2, cur_tot_node
            else:
                tot_node_feature = np.concatenate((tot_node_feature, cur_tot_node_feature), axis=0)
                tot_y1 = np.concatenate((tot_y1, cur_tot_y1), axis=0)
                tot_y2 = np.concatenate((tot_y2, cur_tot_y2), axis=0)
                tot_node = np.concatenate((tot_node, cur_tot_node), axis=0)

        print('node_feature shape is {}'.format(tot_node_feature.shape)) 
        print('label1:pos num {}/{}'.format(tot_y1.sum(),node_size*len(tot_y1)))
        print('label2:pos num {}/{}'.format(tot_y2.sum(),node_size*len(tot_y2)))
        print('label_node:pos num {}/{}'.format(tot_node.sum(),len(tot_node)))
        print("")
            
        for i in range(len(tot_node_feature)):
            self.data.append([tot_node_feature[i,:,:], tot_y1[i,:], tot_y2[i,:], tot_node[i]])

    def __getitem__(self, item):

        return self.data[item]

    def __len__(self):

        return len(self.data)

def collate_fn(batch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    node_array_batch = torch.as_tensor(np.array([node_array for node_array, label1, label2, label_device in batch])).float().to(device)
    label1_batch = torch.as_tensor(np.array([label1 for node_array, label1, label2,  label_device in batch]), dtype=torch.long).to(device)
    label2_batch = torch.as_tensor(np.array([label2 for node_array, label1, label2,  label_device in batch]), dtype=torch.long).to(device)
    label_device = torch.as_tensor(np.array([label_device for node_array, label1, label2, label_device in batch]), dtype=torch.long).to(device)
    
    return node_array_batch, label1_batch, label2_batch, label_device

if __name__ == '__main__':
    train_list = [
        "part-00000-tid-3100902068733874493-703995b0-464e-4b95-a5da-678ae855abd8-9247552-1-c000",
        "part-00000-tid-7484145954299545376-20250523-a161-4255-a085-318a0e30448a-9085938-1-c000",
        "part-00000-tid-6037883635269097681-f66a908a-f064-4437-8f27-d421ec0a7ce3-9114077-1-c000",
        "part-00000-tid-7018722762007920478-9af94004-777c-4719-96f5-d8f33febe320-9125376-1-c000",
        "part-00000-tid-1474096887651793722-aec81a22-91e0-4431-a1a9-849b95078856-9137760-1-c000",
        "part-00000-tid-3023261965903963734-9f51b237-e291-4544-9fd4-066eee729cdb-9155350-1-c000",
        "part-00000-tid-7741923366318972429-e3c02aac-241e-462d-8fbf-fc979e348e21-9183777-1-c000",
        "part-00000-tid-8839355570434301061-eb0d9023-d5c7-454f-9065-cee8eb1967ee-9215572-1-c000"
    ]
    test_list = [
        "part-00000-tid-5972919212486441012-b64810ab-906a-47a1-820e-811fd7b653ac-9226643-1-c000",
        "part-00000-tid-5025539390035129340-c96f04a0-6dce-42a7-b0db-bcba46d81017-9235944-1-c000",
        "part-00000-tid-7879841418289442352-f3f86498-e323-468f-a9b7-66e70242f566-9244120-1-c000"
    ]
    data = PreDataSet(train_list + test_list)
    print(len(data.node_name_list))
    f = open('tot_node_name.txt','w')
    for item in data.node_name_list:
        f.write(item+'\n')