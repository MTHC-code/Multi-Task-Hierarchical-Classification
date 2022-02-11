# Introduction 
This repository contains the code for the **MTHC(Multi-Task Hierarchical Classification)** approach for disk failure prediction. MTHC consists of two main modules: **Multi-Task Module** and **Hierarchical-aware Module**. Compared to existing approaches which do not focus on specific failure types, we emphasis that different teams focuses on different types of disk failures in real practice. Multi-Task Module integrates disks from each team and utilize multi-task learning to enhance the performance for each single task. Moreover, Hierarchical-aware Module introduces a novel node hierarchical classification approach to deal with the extreme data imbalance of disk failure prediction problem.

# Setup
The specific installation environment can refer to ``requirenments.txt``

# Public Dataset
* [Backblaze Dataset](https://www.backblaze.com/b2/hard-drive-test-data.html)
Backblaze take a snapshot of each operational hard drive. This snapshot includes basic drive information along with the S.M.A.R.T. statistics reported by that drive. The daily snapshot of one drive is one record or row of data. All of the drive snapshots for a given day are collected into a file consisting of a row for each active hard drive.
* [Ali Dataset](https://tianchi.aliyun.com/competition/entrance/231775/rankingList/1)
Ali Dataset is a real industrial data collected by the Alibaba Cloud’s data centers widely used to evaluate the performance of methods for disk failure prediction. The public Ali data contains the timestamp, serial number, disk manufacturer, disk model, normalized SMART attributes, raw SMART attributes and fault type of each disk.

# Build and Test
* data preprocessing
    * dataset_node.py : remove empty columns ,regularize data and filter data.

* model
    * single_model.py : train model for each task 
    * multi_model.py  : train multi-task model based on only Multi-Task Module
    * multi_model_node.py : train MTHC model based on  Multi-Task Module and Hierarchy-aware Module

* train and test
    * multi_task_node.py : train different models with MTHC appoarch.

# Training Model
- LSTM
- RNN
- Trans
- TCNN