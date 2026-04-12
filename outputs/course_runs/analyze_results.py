import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_lr_sweep(output_dir='outputs/course_runs'):
    lrs = ['1e-1', '1e-2', '1e-3', '1e-4']
    plt.figure(figsize=(10, 6))
    
    for lr in lrs:
        run_name = f'lr_{lr}_both'
        csv_path = os.path.join(output_dir, run_name, 'history.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            plt.plot(df['epoch'], df['train_loss'], label=f'LR={lr}')
            
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss (Log Scale)')
    plt.title('Learning Rate Sweep - Train Loss (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(os.path.join(output_dir, 'lr_sweep_loss_log.png'))
    plt.close()
    print(f"Saved optimized LR sweep plot to {os.path.join(output_dir, 'lr_sweep_loss_log.png')}")

def generate_ablation_table(output_dir='outputs/course_runs'):
    runs = {
        'baseline_eeg': {'eeg': '✓', 'spec': '', 'fusion': '', 'label': 'EEG Only'},
        'baseline_spec': {'eeg': '', 'spec': '✓', 'fusion': '', 'label': 'Spec Only'},
        'baseline_both': {'eeg': '✓', 'spec': '✓', 'fusion': '✓', 'label': 'GatedFusion'}
    }
    
    data = []
    for run_name, info in runs.items():
        csv_path = os.path.join(output_dir, run_name, 'history.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            best_val_loss = df['val_loss'].min()
            best_val_acc = df['val_acc'].max()
            data.append({
                'EEGResNet1D': info['eeg'],
                'SpectrogramEfficientNet': info['spec'],
                'GatedFusion': info['fusion'],
                'Val Loss': f"{best_val_loss:.4f}",
                'Val Accuracy': f"{best_val_acc:.4f}"
            })
            
    df_ablation = pd.DataFrame(data)
    csv_out = os.path.join(output_dir, 'ablation_study.csv')
    df_ablation.to_csv(csv_out, index=False)
    print(f"Saved ablation study table to {csv_out}")
    
    # Generate LaTeX code
    latex_code = df_ablation.to_latex(index=False, caption="Ablation Study Results", label="tab:ablation", column_format="ccc|cc")
    with open(os.path.join(output_dir, 'ablation_study.tex'), 'w') as f:
        f.write(latex_code)
    print(f"Saved ablation study LaTeX code to {os.path.join(output_dir, 'ablation_study.tex')}")
    return df_ablation

def plot_performance_comparison(output_dir='outputs/course_runs'):
    runs = ['baseline_eeg', 'baseline_spec', 'baseline_both']
    labels = ['EEG Only', 'Spec Only', 'GatedFusion']
    val_losses = []
    val_accs = []
    
    for run in runs:
        csv_path = os.path.join(output_dir, run, 'history.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            val_losses.append(df['val_loss'].min())
            val_accs.append(df['val_acc'].max())
            
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = range(len(labels))
    width = 0.35
    
    rects1 = ax1.bar([i - width/2 for i in x], val_losses, width, label='Val Loss', color='skyblue')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Performance Comparison: EEG vs Spec vs Both')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar([i + width/2 for i in x], val_accs, width, label='Val Accuracy', color='salmon')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_ylim(0, 1.0)
    
    fig.tight_layout()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.savefig(os.path.join(output_dir, 'baseline_performance_comparison.png'))
    plt.close()
    print(f"Saved performance comparison chart to {os.path.join(output_dir, 'baseline_performance_comparison.png')}")

if __name__ == "__main__":
    plot_lr_sweep()
    generate_ablation_table()
    plot_performance_comparison()
