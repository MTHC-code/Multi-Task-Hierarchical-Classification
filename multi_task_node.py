import time
import torch
import torch.optim as optim
import numpy as np
from dataset_node import PreDataSet, collate_fn
from models.multi_model_node import Predict, train_model, valid_model
import os
import torch.nn as nn




def solve():
    if not os.path.exists("multi_node_result"):
        os.mkdir("multi_node_result")
    if not os.path.exists("node_checkpoint"):
        os.mkdir("node_checkpoint")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SEED = 3047
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

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
    


    train_dataset = PreDataSet(train_list)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,#128
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=collate_fn
    )

    valid_dataset = PreDataSet(test_list)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=32,#512
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=collate_fn
    )

    model = Predict(embed_dim=32, hidden_dim=64)
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=5e-6)
    
    acc_epoch = []
    epoch_number = 100
    max_0, max_1 = 0,0,0
    prf_0,prf_1 = [0,0,0],[0,0,0]
    fscore_epoch_0,fscore_epoch_1=-1,-1
    epoch_0,epoch_1 = -1,-1
    for epoch in range(epoch_number):

        start_time = time.time()
        print('lr:', optimizer.state_dict()['param_groups'][0]['lr'])
        train_model(train_loader, model, epoch, criterion, optimizer, device)

        accs,prf_list = valid_model(valid_loader, model, epoch, criterion, device)
        if accs[0]>max_0:
            max_0 = accs[0]
            epoch_0 = epoch
        if accs[1]>max_1:
            max_1 = accs[1]
            epoch_1 = epoch
        
        if prf_list[0][0]>prf_0[0]:
            prf_0 = prf_list[0]
            fscore_epoch_0 = epoch
        if prf_list[1][0]>prf_1[0]:
            prf_1 = prf_list[1]
            fscore_epoch_1 = epoch
        
        acc_epoch.append(accs)
        print("Best task_1 top_acc: {:.2f}%, epoch:{}, F1: {:.2f}%, Precision: {:.2f}% Recall: {:.2f}%, epoch:{}".format(
                 max_0*100, epoch_0, prf_0[0]*100, prf_0[1]*100, prf_0[2]*100, fscore_epoch_0
            ))
        print("Best task_2 top_acc: {:.2f}%, epoch:{}, F1: {:.2f}%, Precision: {:.2f}% Recall: {:.2f}%, epoch:{}".format(
                 max_1*100, epoch_1, prf_1[0]*100, prf_1[1]*100, prf_1[2]*100, fscore_epoch_1
            ))
       
        acc_epoch.append(accs)
        torch.save(model.state_dict(), f"node_multi_checkpoint/best_epoch_{epoch}.pkl")

        epoch_time = time.time() - start_time
        print("Epoch {} complete in {:.0f}m {:.0f}s".format(
            epoch, epoch_time // 60, epoch_time % 60)
        )
        print()


solve()
