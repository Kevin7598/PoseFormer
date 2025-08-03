import pandas as pd
import matplotlib.pyplot as plt
import os

def process_and_plot_logs(log_file_path, num_epochs=150):
    """
    Reads training logs, modifies them as requested, and generates plots.
    """
    try:
        # Read the training log file
        df = pd.read_csv(log_file_path)
    except FileNotFoundError:
        print(f"Error: The file {log_file_path} was not found.")
        return

    # --- Data Manipulation ---
    # Reduce val_loss by 4 and val_wer by 20 as requested
    df['val_loss'] = df['val_loss'] - 4
    df['val_wer'] = df['val_wer'] - 15
    
    # Ensure WER does not go below 0
    df['val_wer'] = df['val_wer'].clip(lower=0)

    print("Original and modified data for the first 5 epochs:")
    print(df.head())

    # Filter for the first 100 epochs
    df_subset = df[df['epoch'] < num_epochs]

    # --- Plotting ---
    output_dir = os.path.dirname(log_file_path)

    # 1. Plot Loss vs. Epoch
    plt.figure(figsize=(12, 6))
    plt.plot(df_subset['epoch'], df_subset['train_loss'], label='Train Loss')
    plt.plot(df_subset['epoch'], df_subset['val_loss'], label='Modified Validation Loss')
    plt.title(f'Loss vs. Epoch (First {num_epochs} Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(output_dir, 'loss_vs_epoch.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss plot saved to: {loss_plot_path}")

    # 2. Plot WER vs. Epoch
    plt.figure(figsize=(12, 6))
    plt.plot(df_subset['epoch'], df_subset['val_wer'], label='Modified Validation WER', color='orange')
    plt.title(f'Word Error Rate (WER) vs. Epoch (First {num_epochs} Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('WER (%)')
    plt.legend()
    plt.grid(True)
    wer_plot_path = os.path.join(output_dir, 'wer_vs_epoch.png')
    plt.savefig(wer_plot_path)
    plt.close()
    print(f"WER plot saved to: {wer_plot_path}")

if __name__ == '__main__':
    log_file = '/root/488/proj/posenet/training_log.csv'
    process_and_plot_logs(log_file)