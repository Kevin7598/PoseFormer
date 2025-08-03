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
    # 读入标签路径
    trainLabelPath = config["trainLabelPath"]
    validLabelPath = config["validLabelPath"]
    testLabelPath = config["testLabelPath"]
    bestModuleSavePath = config["bestModuleSavePath"]
    save_dir = os.path.dirname(bestModuleSavePath)
    os.makedirs(save_dir, exist_ok=True)
    currentModuleSavePath = config["currentModuleSavePath"]
    # device = config["device"]
    inputSize = int(config["inputSize"])
    hiddenSize = int(config["hiddenSize"])
    lr = float(config["lr"])
    batchSize = int(config["batchSize"])
    numWorkers = int(config["numWorkers"])
    pinmMemory = bool(int(config["pinmMemory"]))
    moduleChoice = config["moduleChoice"]
    dataSetName = config["dataSetName"]

    # Set device for multi-GPU training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 数据加载
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

    model = PoseTransformerV2(
        num_classes=vocab_size + 1,   # Number of output classes
        num_joints=84,                # Number of joints in your data
        in_chans=3,                   # Input channels (e.g., x, y, z)
        num_heads=8,                  # Number of attention heads (increased)
        mlp_ratio=1.5,                 # MLP hidden dim ratio (increased)
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.25,                # Added dropout
        attn_drop_rate=0.25,           # Added attention dropout
        drop_path_rate=0.25            # Increased stochastic depth
    )

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    
    # Optimizer optimized for PoseFormer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8)
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    max_grad_norm = 1.0

    # Mixed precision training for PoseFormer
    scaler = GradScaler()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    
    decoder = decode.Decode(word2idx, vocab_size + 1, 'beam')

    best_wer = float('inf')
    train_losses = []
    val_losses = []
    val_wers = []
    lr_list = []
    global_step = 0

    for epoch in range(1, 100):  # More epochs for PoseFormer
        model.train()
        epoch_loss = 0
        batch_idx = 0
        
        for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch}"):
            data, label, lengths = batch["pose"].to(device), batch["label"], batch["poseLength"]
            target = [torch.tensor(l).to(device) for l in label]
            target_lens = torch.tensor([len(t) for t in target])
            target = torch.cat(target)

            with autocast():
                # PoseFormer forward pass
                # data shape: (B, T, num_joints, channels) - already correct for PoseFormer
                logits = model(data, lengths)  # Returns (B, 1, num_classes)
                
                # Reshape for CTC loss: (T, B, C) where T=1 for PoseFormer
                log_probs = logits.permute(1, 0, 2)  # (1, B, num_classes)
                
                # Prepare lengths for CTC
                input_lengths = torch.full((data.size(0),), logits.size(1), dtype=torch.int32, device=device)
                target_lengths = torch.tensor(target_lens, dtype=torch.int32, device=device)

                target = target.to(device)  # Flatten target for CTC

                loss = ctc_loss_fn(log_probs, target, input_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Skipping batch due to nan/inf loss: {loss.item()}")
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            # scheduler.step()

            epoch_loss += loss.item()
            batch_idx += 1
            
        train_losses.append(epoch_loss / len(train_loader))
        scheduler.step()  # Step the scheduler after each epoch
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

                # PoseFormer validation - fix the output handling
                logits = model(data, lengths)  
                # print(f"Debug: logits shape = {logits.shape}")  # Add this to check actual shape
                
                # Handle different possible output shapes
                if len(logits.shape) == 3:  # (B, T, num_classes)
                    log_probs = logits.permute(1, 0, 2)  # (T, B, num_classes)
                    # Use actual sequence lengths, not fixed 1
                    input_lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
                else:  # Handle other shapes appropriately
                    # Adjust based on actual PoseFormer output
                    log_probs = logits
                    input_lengths = torch.ones(data.size(0), dtype=torch.int32, device=device)
                    
                target_lengths = torch.tensor(target_lens, dtype=torch.int32, device=device)
                
                # Add debugging
                # print(f"Debug: log_probs shape = {log_probs.shape}")
                # print(f"Debug: input_lengths = {input_lengths}")
                # print(f"Debug: target_lengths = {target_lengths}")
                # print(f"Debug: target shape = {target.shape}")
                
                loss = ctc_loss_fn(log_probs, target, input_lengths, target_lengths)
                
                # Check if loss is valid
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() == 0:
                    print(f"Warning: Invalid loss = {loss.item()}")
                    continue
                    
                val_loss += loss.item()
                
                # WER calculation (simplified for debugging)
                try:
                    preds, gt = decoder.decode(log_probs, lengths, batch_first=False, probs=False)
                    total_wer += WerScore([gt], targetData, idx2word, 1)
                except Exception as e:
                    print(f"Error in WER calculation: {e}")
                    total_wer += 100.0  # Add penalty for failed decode
        avg_wer = total_wer / len(valid_loader)
        print(f"Epoch {epoch} Val Loss: {val_loss / len(valid_loader):.4f}, WER: {avg_wer:.2f}")
        val_losses.append(val_loss / len(valid_loader))
        val_wers.append(avg_wer)
        
        current_lr = optimizer.param_groups[0]['lr']
        lr_list.append(current_lr)
        
        if avg_wer < best_wer:
            best_wer = avg_wer
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), bestModuleSavePath)
            else:
                torch.save(model.state_dict(), bestModuleSavePath)
            print(f"Best model saved with WER: {best_wer:.2f}")
            
        # Early stopping
        if len(val_wers) > 50 and avg_wer > min(val_wers[-50:]) + 5.0:
            print("Early stopping triggered")
            break
            
        # Save training log
        if epoch % 10 == 0:
            torch.save({
                'epoch': list(range(len(train_losses))),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_wers': val_wers,
                'lr': lr_list
            }, 'poseformer_training_log.pt')
            
            log_data = {
                'epoch': list(range(len(train_losses))),
                'train_loss': train_losses,
                'val_loss': val_losses,
                'val_wer': val_wers,
                'lr': lr_list
            }

            df = pd.DataFrame(log_data)
            df.to_csv('poseformer_training_log.csv', index=False)
    
    # Final save
    log_data = {
        'epoch': list(range(len(train_losses))),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_wer': val_wers,
        'lr': lr_list
    }
    df = pd.DataFrame(log_data)
    df.to_csv('poseformer_training_log.csv', index=False)
    save_plots(df, save_dir="poseformer_plots")


if __name__ == '__main__':
    seed_everything(42)
    config = readConfig()
    # Ensure we're using the right module choice
    config["moduleChoice"] = "poseformer"
    train_model(config)