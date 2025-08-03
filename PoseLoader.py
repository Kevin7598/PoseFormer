import os
import numpy as np
import torch
from torch.utils.data import Dataset
import csv
import torch.nn as nn
import torch.nn.functional as F

PAD = ' '

def PreWords(words):
    for i in range(len(words)):
        word = words[i]

        n = 0
        subFlag = False
        wordList = list(word)
        for j in range(len(word)):
            if word[j] == "(" or word[j] == "{" or word[j] == "[" or word[j] == "（":
                subFlag = True

            if subFlag:
                wordList.pop(j - n)
                n = n + 1

            if word[j] == ")" or word[j] == "}" or word[j] == "]" or word[j] == "）":
                subFlag = False

        word = "".join(wordList)

        if word[-1].isdigit():
            if not word[0].isdigit():
                wordList = list(word)
                wordList.pop(len(word) - 1)
                word = "".join(wordList)

        if word[0] == "," or word[0] == "，":
            wordList = list(word)
            wordList[0] = '，'
            word = ''.join(wordList)

        if word[0] == "?" or word[0] == "？":
            wordList = list(word)
            wordList[0] = '？'
            word = ''.join(wordList)

        if word.isdigit():
            word = str(int(word))

        words[i] = word

    return words

def Word2Id(trainLabelPath, validLabelPath, testLabelPath):
    wordList = []
    with open(trainLabelPath, 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        for n, row in enumerate(reader):
            if n != 0:
                words = row[3].split("/")
                words = PreWords(words)
                wordList += words

    with open(validLabelPath, 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        for n, row in enumerate(reader):
            if n != 0:
                words = row[3].split("/")
                words = PreWords(words)
                wordList += words

    with open(testLabelPath, 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        for n, row in enumerate(reader):
            if n != 0:
                words = row[3].split("/")
                words = PreWords(words)
                wordList += words

    idx2word = [PAD]
    set2list = sorted(list(set(wordList)))
    idx2word.extend(set2list)

    word2idx = {w: i for i, w in enumerate(idx2word)}

    return word2idx, len(idx2word) - 1, idx2word

class MyDataset(Dataset):
    def __init__(self, dataPath, labelPath, word2idx, isTrain=True, transform=None):
        self.dataPath = dataPath
        self.labelPath = labelPath
        self.word2idx = word2idx
        self.transform = transform
        self.isTrain = isTrain

        lableDict = {}
        with open(labelPath, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for n, row in enumerate(reader):
                if n != 0:
                    lableDict[row[0]] = row[3]

        lable = {}
        for line in lableDict:
            sentences = lableDict[line].split("/")
            sentences = PreWords(sentences)

            txtInt = []
            for i in sentences:
                txtInt.append(word2idx[i])

            lable[line] = txtInt

        fileNames = sorted(os.listdir(dataPath))

        poses = []
        for name in fileNames:
            fileName = os.path.join(dataPath, name)
            poseNames = sorted(os.listdir(fileName))
            for poseName in poseNames:
                poseNumber = os.path.splitext(poseName)[0]
                try:
                    poseSeqPath = os.path.join(dataPath, name, poseName)
                    poses.append((poseSeqPath, lable[poseNumber]))
                except:
                    print(poseName)


        self.poses = poses

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        posePath, label = self.poses[idx]
        info = posePath.split("/")[-1]
        pose = np.load(posePath, allow_pickle = True)  # shape: [F, J, D]

        pose = torch.tensor(pose, dtype=torch.float32)  # [F, J, D]
        sample = {
            "pose": pose,
            "poseLength": pose.shape[0],
            "label": label,
            "info": info
        }
        return sample
    
def collate_fn(batch):
    batch.sort(key=lambda x: x["poseLength"], reverse=True)
    maxLen = batch[0]["poseLength"]
    B = len(batch)
    poseDim = batch[0]["pose"].shape[1:]

    paddedPose = torch.zeros((B, maxLen) + poseDim, dtype=torch.float32)
    poseLens = []
    labels = []

    for i, sample in enumerate(batch):
        T = sample["pose"].shape[0]
        paddedPose[i, :T] = sample["pose"]
        poseLens.append(T)
        labels.append(sample["label"])

    return {
        "pose": paddedPose,
        "poseLength": torch.tensor(poseLens),
        "label": labels
    }

class SeqKD(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, T=1):
        super(SeqKD, self).__init__()
        self.kdloss = nn.KLDivLoss(reduction='batchmean')
        self.T = T

    def forward(self, prediction_logits, ref_logits, use_blank=True):
        start_idx = 0 if use_blank else 1
        prediction_logits = F.log_softmax(prediction_logits[:, :, start_idx:]/self.T, dim=-1) \
            .view(-1, ref_logits.shape[2] - start_idx)
        ref_probs = F.softmax(ref_logits[:, :, start_idx:]/self.T, dim=-1) \
            .view(-1, ref_logits.shape[2] - start_idx)
        loss = self.kdloss(prediction_logits, ref_probs)*self.T*self.T
        return loss
