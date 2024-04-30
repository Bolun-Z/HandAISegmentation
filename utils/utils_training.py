import os
import random
import numpy as np
from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dataset(dataset_path, train_ratio=0.9):
    file_lines = []
    image_root = os.path.join(dataset_path, 'image')
    files = os.listdir(image_root)

    for file in files:
        if file.endswith(".nii.gz"):
            file_lines.append(file)

    random.seed(1)
    shuffle_index = np.arange(len(file_lines), dtype=np.int32)
    shuffle(shuffle_index)
    file_lines = np.array(file_lines, dtype=np.object)
    file_lines = file_lines[shuffle_index]

    # Divide the train and val dataset
    num_train = int(len(file_lines) * train_ratio)
    val_lines = file_lines[num_train:]
    train_lines = file_lines[:num_train]

    return train_lines, val_lines

class FineGrained_feat(torch.nn.Module):
    def __init__(self, batch_size):
        super(FineGrained_feat, self).__init__()
        self.batch_size = batch_size

        self.ignore_label = 0 # ignored class
        self.max_samples = 1024
        # self.max_samples = 512
        # self.max_samples = 2048

    def soft_dilate(self, input_tensor):
        if len(input_tensor.shape) == 4:
            return F.max_pool2d(input_tensor, (3, 3), (1, 1), (1, 1))
        elif len(input_tensor.shape) == 5:
            input_tensor = F.max_pool3d(input_tensor, (3, 3, 3), (1, 1, 1), (1, 1, 1))
            return F.max_pool3d(input_tensor, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def _anchor_sampling(self, feat_map, labels, preds):
        batch_size = labels.shape[0]
        n_channel = feat_map.shape[1]
        classes = []
        total_classes = 0

        preds = torch.argmax(torch.softmax(preds, dim=1), dim=1).unsqueeze(1).float()
        preds_d = self.soft_dilate(preds)

        # calculate the total number of classes in the batch
        for b in range(batch_size):
            batch_classes = torch.unique(labels[b])
            batch_classes = [x for x in batch_classes if x != self.ignore_label]
            classes.append(batch_classes)
            total_classes += len(batch_classes)

        feat_map_extract = torch.zeros((total_classes, n_channel, self.max_samples), dtype=torch.float).cuda()
        feat_label = torch.zeros(total_classes, dtype=torch.float).cuda()

        feat_map_extract_hard_keep = torch.zeros((total_classes, n_channel, self.max_samples // 2), dtype=torch.float).cuda()
        feat_map_extract_rest_keep = torch.zeros((total_classes, n_channel, self.max_samples // 2), dtype=torch.float).cuda()

        ## Extract fine-grained feature from each class region
        i = 0
        for b in range(batch_size):
            _preds_d = preds_d[b].view(-1)
            _feat_map = feat_map[b].view(n_channel, -1)
            _labels = labels[b].view(-1)
            batch_classes = classes[b]
            for cls in batch_classes:
                hard_indices = ((_preds_d != cls) & (_labels == cls)).nonzero()
                easy_indices = ((_preds_d == cls) & (_labels == cls)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                # ## Strategy 1：choose hard and random samples from all samples
                # perm = torch.randperm(num_hard)
                # # hard_indices = hard_indices[perm[:num_hard_keep]]
                # hard_indices = hard_indices[perm]
                # perm = torch.randperm(num_easy)
                # # easy_indices = easy_indices[perm[:num_easy_keep]]
                # easy_indices = easy_indices[perm]
                # indices = torch.cat((hard_indices, easy_indices), dim=0)
                #
                # ### 筛选出目标对象的特征，但是维度大小不固定
                # # index = torch.nonzero(_labels == cls, as_tuple=False)
                # feat_map_extract_t = torch.index_select(_feat_map, 1, indices.squeeze())
                # num_feat = feat_map_extract_t.shape[1]
                # perm = torch.randperm(num_feat)
                # feat_map_extract_t_keep = feat_map_extract_t[:, perm[:self.max_samples]]
                # feat_map_extract[i][:, :feat_map_extract_t_keep.shape[1]] = feat_map_extract_t_keep
                # feat_label[i] = cls
                # i = i + 1

                ## Strategy 2: The half of the samples are hard samples, and the rest are random samples
                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm]
                num_hard_keep = self.max_samples // 2
                indices_hard = hard_indices[perm[:num_hard_keep]]
                hard_indices_random = hard_indices[perm[num_hard_keep:]]

                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm]
                indices_rest = torch.cat((hard_indices_random, easy_indices), dim=0)

                num_rest = indices_rest.shape[0]
                perm = torch.randperm(num_rest)
                num_rest_keep = self.max_samples // 2
                indices_rest = indices_rest[perm[:num_rest_keep]]

                ### Get the features of the target class
                feat_map_extract_hard = torch.index_select(_feat_map, 1, indices_hard.squeeze())
                feat_map_extract_hard_keep[i][:, :feat_map_extract_hard.shape[1]] = feat_map_extract_hard

                feat_map_extract_rest = torch.index_select(_feat_map, 1, indices_rest.squeeze())
                feat_map_extract_rest_keep[i][:, :feat_map_extract_rest.shape[1]] = feat_map_extract_rest
                feat_map_extract[i] = torch.cat([feat_map_extract_hard_keep[i], feat_map_extract_rest_keep[i]], dim=1)
                feat_label[i] = cls
                i = i + 1

        return feat_map_extract, feat_label

class FineGrained_ContrastLoss(torch.nn.Module):
    def __init__(self, batch_size, head='mlp', dim_in=256, feat_dim=64):
        super(FineGrained_ContrastLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = 0.2
        self.base_temperature = 0.07

        self.criterion = nn.CrossEntropyLoss().float()
        self.n_views = 2 # postive and negative
        self.ignore_label = 0
        self.max_samples = 1024

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim).to(device)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Conv3d(dim_in, dim_in, kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(dim_in, feat_dim, kernel_size=1, stride=1)
            ).to(device)
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def soft_dilate(self, input_tensor):
        if len(input_tensor.shape) == 4:
            return F.max_pool2d(input_tensor, (3, 3), (1, 1), (1, 1))
        elif len(input_tensor.shape) == 5:
            input_tensor = F.max_pool3d(input_tensor, (3, 3, 3), (1, 1, 1), (1, 1, 1))
            return F.max_pool3d(input_tensor, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def Info_nce_loss(self, features, feat_label):
        n_views = features.shape[0]
        labels = (feat_label.unsqueeze(0) == feat_label.unsqueeze(1)).float()
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features.float(), features.float().T)

        if similarity_matrix.shape != (n_views, n_views):
            return 0
        elif similarity_matrix.shape != labels.shape:
            return 0

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(-1, 1)  # N * 1
        negatives = similarity_matrix[~labels.bool()] # N * k
        bia = 0
        if positives.shape[0] != 0:
            bia = negatives.shape[0] % positives.shape[0]
            negatives = negatives[:(negatives.shape[0] - bia)].view(positives.shape[0], -1)
        else:
            positives = similarity_matrix[labels.bool()].view(2, 0)  # N * 1
            negatives = negatives[:(negatives.shape[0] - bia)].view(2, -1)

        logits = torch.cat([positives, negatives], dim=1) # N * (1 + k)
        labels = torch.zeros(positives.shape[0], dtype=torch.long).to(device)
        logits = logits / self.temperature

        try:
            loss = self.criterion(logits, labels)
        except:
            print("logits", logits.shape)
            print("labels", labels.shape)
            loss = 0
            loss = torch.as_tensor(loss, dtype=torch.float32)

        return loss

    def forward(self, feat_map_extract, feat_label):
        feat = F.normalize(self.head(feat_map_extract), dim=1).view(feat_map_extract.shape[0], -1)
        loss = self.Info_nce_loss(feat, feat_label)
        return loss

class Dice_loss(nn.Module):
    def __init__(self, n_classes):
        super(Dice_loss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        inputs = inputs.float()
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

if __name__ == "__main__":
    """
    Sample usage. In order to test the code, Input and GT are randomly populated with values.
    """

    # Parameters for creating random input
    num_classes = height = width = depth = 5

    dim = 3

    x = torch.rand(1, num_classes ,depth, height, width)
    y = torch.randint(0, num_classes, (1, depth, height, width))

    Dice_loss
