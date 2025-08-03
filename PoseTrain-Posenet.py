import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import lr_scheduler
import numpy as np
from tqdm import tqdm
import random, os

from PoseFormer import PoseTransformerV2
from PoseNet import PoseNet, PoseLSTM, PoseTransformer
import PoseLoader
import decode
from WER import WerScore
from ReadConfig import readConfig
import pandas as pd
import matplotlib.pyplot as plt

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_plots(log_df, save_dir="plots"):
    """Saves plots for training/validation loss and validation WER."""
    os.makedirs(save_dir, exist_ok=True)

    # Plotting Loss
    plt.figure(figsize=(12, 5))
    plt.plot(log_df['epoch'], log_df['train_loss'], label='Train Loss')
    plt.plot(log_df['val_loss'], label='Validation Loss')
    plt.title('Loss Progress Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_progress.png'))
    plt.close()

    # Plotting WER
    plt.figure(figsize=(12, 5))
    plt.plot(log_df['epoch'], log_df['val_wer'], label='Validation WER', color='orange')
    plt.title('Word Error Rate (WER) Progress Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('WER (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'wer_progress.png'))
    plt.close()

def adjust_learning_rate_with_warmup(optimizer, step, warmup_steps, base_lr, scheduler=None):
    if step < warmup_steps:
        lr = base_lr * (step + 1) / warmup_steps
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        if scheduler:
            scheduler.step()

def train_model(config):
    trainDataPath = config["trainDataPath"]
    validDataPath = config["validDataPath"]
    testDataPath = config["testDataPath"]
    trainLabelPath = config["trainLabelPath"]
    validLabelPath = config["validLabelPath"]
    testLabelPath = config["testLabelPath"]
    bestModuleSavePath = config["bestModuleSavePath"]
    save_dir = os.path.dirname(bestModuleSavePath)
    os.makedirs(save_dir, exist_ok=True)
    currentModuleSavePath = config["currentModuleSavePath"]
    device_ids = [0, 1] # Use GPU 0 and 1
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    inputSize = int(config["inputSize"])
    hiddenSize = int(config["hiddenSize"])
    lr = float(config["lr"])
    batchSize = int(config["batchSize"])
    numWorkers = int(config["numWorkers"])
    pinmMemory = bool(int(config["pinmMemory"]))
    moduleChoice = config["moduleChoice"]
    dataSetName = config["dataSetName"]

    word2idx, vocab_size, idx2word = PoseLoader.Word2Id(trainLabelPath, validLabelPath, testLabelPath)
    train_set = PoseLoader.MyDataset(trainDataPath, trainLabelPath, word2idx, True)
    valid_set = PoseLoader.MyDataset(validDataPath, validLabelPath, word2idx, False)
    
    print(f"Training set size: {len(train_set)}, Validation set size: {len(valid_set)}")

    train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True, 
                              num_workers=numWorkers, pin_memory=pinmMemory,
                              collate_fn=PoseLoader.collate_fn, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, 
                              num_workers=numWorkers, pin_memory=pinmMemory,
                              collate_fn=PoseLoader.collate_fn)

    if moduleChoice == 'transformer':
        model = PoseTransformer(input_dim=inputSize, hidden_dim=hiddenSize, num_classes=vocab_size + 1)
    elif moduleChoice == 'lstm':
        model = PoseLSTM(input_dim=inputSize, hidden_dim=hiddenSize, num_classes=vocab_size + 1)
    elif moduleChoice == 'posenet':
        model = PoseNet(input_dim=inputSize, hidden_dim=hiddenSize, num_classes=vocab_size + 1)
    elif moduleChoice == 'posenetv2':
        model = PoseTransformerV2(
            num_classes=vocab_size + 1,   # Number of output classes
            num_joints=84,                # Number of joints in your data
            in_chans=3,                   # Input channels (e.g., x, y, z)
            num_heads=4,                  # Number of attention heads
            mlp_ratio=2.,                 # MLP hidden dim ratio
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.25,
            attn_drop_rate=0.25,
            drop_path_rate=0.3           # Stochastic depth for regularization
        )
    model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids) # Wrap model for multi-GPU

    # for posenet decay = 0.001 is good
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.98), eps=1e-9)
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    kld = PoseLoader.SeqKD(T=8)
    logSoftMax = nn.LogSoftmax(dim=-1)
    max_grad_norm = 1.0

    # remove for transformer
    scaler = GradScaler()
    warmup_steps = 1000
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-3,                    # Peak learning rate
        epochs=200,                     # Total epochs
        steps_per_epoch=len(train_loader),  # Steps per epoch
        pct_start=0.1,                  # 10% of training for warmup
        anneal_strategy='cos'           # Cosine annealing after peak
    )
    decoder = decode.Decode(word2idx, vocab_size + 1, 'beam')

    best_wer = float('inf')
    train_losses = []
    val_losses = []
    val_wers = []
    lr_list = []
    global_step = 0

    for epoch in range(1, 800):
        model.train()
        epoch_loss = 0
        batch_idx = 0

        for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch}"):
            data, label, lengths = batch["pose"].to(device), batch["label"], batch["poseLength"]
            target = [torch.tensor(l).to(device) for l in label]
            target_lens = torch.tensor([len(t) for t in target])

            # posenet
            target = torch.cat(target)

            # transformer
            # target_cat = torch.cat(target)


            with autocast(enabled=False):
                # transformer
                # B, T, N, D = data.shape  # B, T, 84, 3
                # data = data.view(B, T, N * D)
                # out, out_len = model(data, lengths)
                # # print(f"logProb shape:{out.shape}")
                # log_probs = out.permute(1, 0, 2)
                # input_lengths = torch.tensor(lengths, dtype=torch.int32)
                # target_lengths = torch.tensor(target_lens, dtype=torch.int32)
                # loss = ctc_loss_fn(log_probs, target_cat, input_lengths, target_lengths)
                # log_probs = log_probs.detach().cpu()
                # print("log_probs shape:", log_probs.shape)
                # print("Top class IDs:", log_probs[0].argmax(dim=-1))

                #posenet
                logProbs1, logProbs2, logProbs3, logProbs4, logProbs5, lgt, x1, x2, x3 = model(data, lengths, True)

                loss3 = 25 * kld(logProbs2, logProbs1, use_blank=False)

                logProbs1 = logSoftMax(logProbs1)
                logProbs2 = logSoftMax(logProbs2)

                loss1 = ctc_loss_fn(logProbs1, target, lgt, target_lens).mean()
                loss2 = ctc_loss_fn(logProbs2, target, lgt, target_lens).mean()
                
                loss6 = 25 * kld(logProbs4, logProbs3, use_blank=False)

                logProbs3 = logSoftMax(logProbs3)
                logProbs4 = logSoftMax(logProbs4)

                loss4 = ctc_loss_fn(logProbs3, target, lgt, target_lens).mean()
                loss5 = ctc_loss_fn(logProbs4, target, lgt, target_lens).mean()

                logProbs5 = logSoftMax(logProbs5)
                loss7 = ctc_loss_fn(logProbs5, target, lgt, target_lens).mean()

                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()

            # posenet
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            # transformer
            # loss.backward()
            # optimizer.step()

            epoch_loss += loss.item()
            
            batch_idx=batch_idx+1
            # if batch_idx % 10 == 0:
            #     print(f"[Epoch {epoch}] Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
        train_losses.append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch} Train Loss: {epoch_loss / len(train_loader):.4f}")

        # 验证
        model.eval()
        val_loss, total_wer = 0, 0

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="[Valid]"):
                data, label, lengths = batch["pose"].to(device), batch["label"], batch["poseLength"]
                targetData = [torch.tensor(l).to(device) for l in label]
                target_lens = torch.tensor([len(t) for t in targetData])
                target = torch.cat(targetData)

                # transformer
                # B, T, N, D = data.shape  # B, T, 84, 3
                # data = data.view(B, T, N * D)
                # out, out_len = model(data, lengths)
                # log_probs = out.permute(1, 0, 2)
                # input_lengths = torch.tensor(lengths, dtype=torch.int32)
                # target_lengths = torch.tensor(target_lens, dtype=torch.int32)
                # loss = ctc_loss_fn(log_probs, target, input_lengths, target_lengths)
                
                # val_loss += loss.item()

                # # Decode predictions for WER calculation
                # preds, gt = decoder.decode(log_probs, lengths, batch_first=False, probs=False)
                # total_wer += WerScore([gt], targetData, idx2word, 1)


                # posenet
                logProbs1, logProbs2, logProbs3, logProbs4, logProbs5, lgt, x1, x2, x3 = model(data, lengths, True)

                loss3 = 25 * kld(logProbs2, logProbs1, use_blank=False)

                logProbs1 = logSoftMax(logProbs1)
                logProbs2 = logSoftMax(logProbs2)

                loss1 = ctc_loss_fn(logProbs1, target, lgt, target_lens).mean()
                loss2 = ctc_loss_fn(logProbs2, target, lgt, target_lens).mean()
                
                loss6 = 25 * kld(logProbs4, logProbs3, use_blank=False)

                logProbs3 = logSoftMax(logProbs3)
                logProbs4 = logSoftMax(logProbs4)

                loss4 = ctc_loss_fn(logProbs3, target, lgt, target_lens).mean()
                loss5 = ctc_loss_fn(logProbs4, target, lgt, target_lens).mean()

                logProbs5 = logSoftMax(logProbs5)
                loss7 = ctc_loss_fn(logProbs5, target, lgt, target_lens).mean()

                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7

                val_loss += loss.item()

                preds, gt = decoder.decode(logProbs1, lgt, batch_first=False, probs=False)
                total_wer += WerScore([gt], targetData, idx2word, 1)

        avg_wer = total_wer / len(valid_loader)
        print(f"Epoch {epoch} Val Loss: {val_loss / len(valid_loader):.4f}, WER: {avg_wer:.2f}")
        val_losses.append(val_loss / len(valid_loader))
        val_wers.append(avg_wer)
        adjust_learning_rate_with_warmup(optimizer, global_step, 20, lr, scheduler)
        global_step += 1
        lr_list.append(optimizer.param_groups[0]['lr'])
        if avg_wer < best_wer:
            best_wer = avg_wer
            torch.save(model.state_dict(), bestModuleSavePath)
            print(f"Best model saved with WER: {best_wer:.2f}")
        if (epoch + 1) % 1 == 0:
            torch.save({
                'epoch': list(range(len(train_losses))),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_wers': val_wers,
                'lr': lr_list
            }, 'training_log.pt')
            log_data = {
                'epoch': list(range(len(train_losses))),
                'train_loss': train_losses,
                'val_loss': val_losses,
                'val_wer': val_wers,
                'lr': lr_list
            }

            df = pd.DataFrame(log_data)
            df.to_csv('training_log.csv', index=False)
    save_plots(df)


if __name__ == '__main__':
    seed_everything(10)
    config = readConfig()
    train_model(config)
