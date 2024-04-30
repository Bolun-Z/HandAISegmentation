import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.callbacks import LossHistory
from utils.CreatDataset import CreatDataset
from utils.utils_training import Dice_loss, FineGrained_ContrastLoss, FineGrained_feat, load_dataset
from utils.training_set import set_optimizer_lr, get_lr_scheduler, set_optimizer_lr_poly
from networks.AMWNet import AMWNet
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from argparse import ArgumentParser

# =========================Train parameters=======================
torch.manual_seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser(description='Process some integers.')
parser.add_argument('--batch_size', default=2, type=int, help='The batch size of training samples')
parser.add_argument('--epochs', default=100, type=int, help='The training epoch')
parser.add_argument('--Cuda', default=True, type=bool, help='Use cuda or not')
parser.add_argument('--Multi_GPU', default=True, type=bool, help='Use multi gpu or not')
parser.add_argument('--AMP', default=True, type=bool, help='Use automatic mixed precision gpu or not')
parser.add_argument('--save_period', default=3, type=int, help='The saving period')
parser.add_argument('--model', default='AMFNet', type=str, help='The network')
parser.add_argument('--FCL', default=False, type=bool, help='Using fine-grained contrastive leaning')

parser.add_argument('--Init_lr', default=1e-3, type=float, help='The initial learning rate')
parser.add_argument('--Min_lr', default=1e-5, type=float, help='The minimum learning rate')
parser.add_argument('--lr_decay_type', default='cos', type=str, help='The learning rate drop manner (cos or step)')

parser.add_argument('--num_classes', default=1 + 38, type=int, help='Number of label classes (38+1 or 1+1)')
parser.add_argument('--input_shape', default=[16, 224, 448], type=int,
                   help='Input all image dimensions (d h w, e.g. --input-size [16, 256, 512], uses model default if empty')

parser.add_argument('--Dataset_path', default='../Dataset/Train/86cases/16_256_512_downsample', type=str, help='The path root of dataset')
parser.add_argument('--pretrained_model', default='', type=str, help='The path of pretrained model')
parser.add_argument('--save_dir', default='logs/', type=str, help='The path of saving training history')
parser.add_argument('--output_path', default='logs/', type=str, help='The path of logs')

parser.add_argument('--optimizer_type', default='adam', type=str, help='The optimizer type (adam or sgd)')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum of the optimizer')
parser.add_argument('--weight_decay', default=0, type=float, help='The weight decay (0 when using adam)')

args = parser.parse_args()

def define_model():
    model = AMWNet(n_classes=args.num_classes)
    if args.Multi_GPU:
        model = nn.DataParallel(model)
    return model

def get_dataloader():
    train_lines, val_lines = load_dataset(dataset_path=args.Dataset_path, train_ratio=0.9)
    num_train = len(train_lines)
    num_val = len(val_lines)

    epoch_step = num_train // args.batch_size
    epoch_step_val = num_val // args.batch_size

    train_dataset = CreatDataset(train_lines, args.input_shape, args.num_classes, args.Dataset_path, mode='Train')
    val_dataset = CreatDataset(val_lines, args.input_shape, args.num_classes, args.Dataset_path, mode='Train')

    trainloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=2, pin_memory=True)
    valloader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size, num_workers=2, pin_memory=True)

    return trainloader, valloader, epoch_step, epoch_step_val

def train_eval(trainloader, valloader, epoch_step, epoch_step_val):
    writer = SummaryWriter(log_dir=args.save_dir)
    # =========================model define=======================
    model = define_model()
    model.train()
    model.to(device)

    if args.pretrained_model != '':
        print('Load weights {}.'.format(args.pretrained_model))
        pretrained_dict = torch.load(args.pretrained_model, map_location=device)
        del pretrained_dict['out.conv.conv.weight']
        del pretrained_dict['out.conv.conv.bias']
        model.load_state_dict(pretrained_dict, strict=False)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    print(f'{total_params / (1024 * 1024):.2f}M total parameters.')

    # =========================train strategy=======================
    FG_ContrastLoss = FineGrained_ContrastLoss(batch_size=args.batch_size).to(device)
    finegrained_feat = FineGrained_feat(batch_size=args.batch_size)
    Dice_Loss = Dice_loss(n_classes=args.num_classes)
    if args.AMP:
        scaler = GradScaler()

    nbs = 16
    lr_limit_max = 1e-4 if args.optimizer_type == 'adam' else 1e-1
    lr_limit_min = 1e-4 if args.optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(args.batch_size / nbs * args.Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(args.batch_size / nbs * args.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    optimizer = {
        'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(args.momentum, 0.999),
                           weight_decay=args.weight_decay,
                           eps=1e-3),
        'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=args.momentum, nesterov=True,
                         weight_decay=args.weight_decay)
    }[args.optimizer_type]

    loss_history = LossHistory(args.save_dir, model, input_shape=args.input_shape)
    for epoch in range(args.epochs):
        train_loss = 0
        val_loss = 0
        train_epoch_loss = []
        val_epoch_loss = []

        # lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.epochs)
        # set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        set_optimizer_lr_poly(optimizer=optimizer, epoch=epoch, num_epochs=args.epochs, base_lr=args.Init_lr)

        # =========================train=======================
        for iteration, batch in enumerate(tqdm(trainloader)):
            images, labels = batch
            with torch.no_grad():
                images = images.to(device).float()
                labels = labels.to(device).long()
            optimizer.zero_grad()

            if args.AMP:
                with autocast():
                    # AMWNet with fine-grain contrastive learning
                    ec_feat1, outputs = model(images)
                    feat_map_extract, feat_label = finegrained_feat._anchor_sampling(ec_feat1, labels, outputs)
                    if args.Multi_GPU:
                        feat_map_extract = model.module.encoder(feat_map_extract)  # DataParallel
                    else:
                        feat_map_extract = model.encoder(feat_map_extract)
                    FG_Con_loss = FG_ContrastLoss(feat_map_extract, feat_label)
                    dice_loss = Dice_Loss(outputs, labels, softmax=True)
                    loss_value = 0.3 * FG_Con_loss + dice_loss

                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                ec_feat1, outputs = model(images)
                feat_map_extract, feat_label = finegrained_feat._anchor_sampling(ec_feat1, labels, outputs)
                if args.Multi_GPU:
                    feat_map_extract = model.module.encoder(feat_map_extract)  # DataParallel
                else:
                    feat_map_extract = model.encoder(feat_map_extract)
                FG_Con_loss = FG_ContrastLoss(feat_map_extract, feat_label)
                dice_loss = Dice_Loss(outputs, labels, softmax=True)
                loss_value = dice_loss + FG_Con_loss
                loss_value.backward()
                optimizer.step()

            train_loss += loss_value.item()
            with torch.no_grad():
                train_epoch_loss.append(train_loss)
        print('====>Training: Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / (iteration + 1)))

        # =========================val=======================
        with torch.no_grad():
            model.eval()
            for iteration, batch in enumerate(tqdm(valloader)):
                    images = images.to(device).float()
                    labels = labels.to(device).long()

                    if args.AMP:
                        with autocast():
                            # AMWNet with fine-grain contrastive learning
                            ec_feat1, outputs = model(images)
                            feat_map_extract, feat_label = finegrained_feat._anchor_sampling(ec_feat1, labels, outputs)

                            if args.Multi_GPU:
                                feat_map_extract = model.module.encoder(feat_map_extract)  # DataParallel
                            else:
                                feat_map_extract = model.encoder(feat_map_extract)

                            FG_Con_loss = FG_ContrastLoss(feat_map_extract, feat_label)
                            dice_loss = Dice_Loss(outputs, labels, softmax=True)
                            loss_value = dice_loss + 0.3 * FG_Con_loss

                    else:
                        ec_feat1, outputs = model(images)
                        feat_map_extract, feat_label = finegrained_feat._anchor_sampling(ec_feat1, labels, outputs)
                        if args.Multi_GPU:
                            feat_map_extract = model.module.encoder(feat_map_extract)  # DataParallel
                        else:
                            feat_map_extract = model.encoder(feat_map_extract)
                        dice_loss = Dice_Loss(outputs, labels, softmax=True)
                        FG_Con_loss = FG_ContrastLoss(feat_map_extract, feat_label)
                        loss_value = dice_loss + 0.3 * FG_Con_loss

                    val_loss += loss_value.item()
                    val_epoch_loss.append(val_loss)

            print('====>Validation: Epoch: {} Average loss: {:.4f} '.format(epoch, val_loss / (iteration + 1)))
            loss_history.append_loss(epoch + 1, train_loss / epoch_step, val_loss / epoch_step_val)

        # =========================save model=====================
        if ((epoch + 1) % args.save_period == 0):
            SAVE_PATH = args.output_path + str(epoch + 1) + ".pth"
            print("save_path", SAVE_PATH)
            torch.save(model.state_dict(), SAVE_PATH)

    # =========================Finish training=======================
    SAVE_PATH = args.output_path + "Final_model" + ".pth"
    torch.save(model.state_dict(), SAVE_PATH)
    print("====> Finish training <=====")
    loss_history.writer.close()

if __name__ == '__main__':
    trainloader, valloader, epoch_step, epoch_step_val = get_dataloader()
    train_eval(trainloader, valloader, epoch_step, epoch_step_val)




