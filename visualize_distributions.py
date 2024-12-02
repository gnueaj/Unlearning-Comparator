import json
import numpy as np
import matplotlib.pyplot as plt
import torch

def make_break_marks(ax, y_pos):
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    dx = 0.01
    dy = 0.01
    ax.plot((-dx, +dx), (y_pos-dy, y_pos+dy), **kwargs)
    ax.plot((-dx, +dx), (y_pos-2*dy, y_pos+2*dy), **kwargs)

def calculate_scores(values1, values2, bins, range_vals):
    thresholds = np.linspace(range_vals[0], range_vals[1], bins)
    scores = []
    DELTA = 1e-05
    EPSILON = 1e-10
    
    values1 = np.array(values1)
    values2 = np.array(values2)
    
    # Compare medians to determine positive class
    ga_median = np.median(values1)
    retrain_median = np.median(values2)
    retrain_is_positive = retrain_median > ga_median
    
    pos_confs_ = torch.from_numpy(values2 if retrain_is_positive else values1)
    neg_confs_ = torch.from_numpy(values1 if retrain_is_positive else values2)
    
    for threshold in thresholds:
        threshold_ = torch.tensor(threshold)
        
        tpr = torch.mean((pos_confs_ >= threshold_).float()).item()
        fpr = torch.mean((neg_confs_ >= threshold_).float()).item()
        tnr = torch.mean((neg_confs_ < threshold_).float()).item()
        fnr = torch.mean((pos_confs_ < threshold_).float()).item()
        
        # Modified epsilon calculation
        if fpr == 0 and fnr == 0:
            epsilon = float('inf')
            forgetting_score = 0  # 완벽한 구분 = 완벽히 잊혀짐
        elif fpr >= (1 - DELTA) or fnr >= (1 - DELTA):
            epsilon = 0
            forgetting_score = 1  # 구분 불가 = 전혀 잊혀지지 않음
        else:
            # 안전한 범위로 클리핑
            safe_fpr = np.clip(fpr, EPSILON, 1 - DELTA - EPSILON)
            safe_fnr = np.clip(fnr, EPSILON, 1 - DELTA - EPSILON)
            
            # 수정된 epsilon 계산
            log_ratio1 = np.log(1 - DELTA - safe_fpr) - np.log(safe_fnr)
            log_ratio2 = np.log(1 - DELTA - safe_fnr) - np.log(safe_fpr)
            
            # 두 비율의 균형을 맞춤
            epsilon = max(0, min(log_ratio1, log_ratio2))
            # Compute forgetting score: 2^(-epsilon)
            forgetting_score = 2 ** (-epsilon)
        
        scores.append({
            'threshold': threshold,
            'fpr': fpr,
            'fnr': fnr,
            'epsilon': epsilon,
            'forgetting_score': forgetting_score
        })
    
    return scores

def plot_distributions(ax, values1, values2, bins, range_vals, legend1, legend2, max_display=40):
    # Calculate histograms for both datasets
    counts1, bin_edges = np.histogram(values1, bins=bins, range=range_vals)  # GA (purple)
    counts2, _ = np.histogram(values2, bins=bins, range=range_vals)  # Retrain (gray)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # For each bin
    for x, count1, count2 in zip(bin_centers, counts1, counts2):
        # Plot GA points (purple) starting from y=0
        count1_display = min(count1, max_display)
        for y in range(count1_display):
            ax.plot(x, y, 'o', color='#8B5CF6', alpha=0.7, markersize=6)  # 보라색으로 변경
        if count1 > max_display:
            ax.plot(x, 42, 'o', color='#8B5CF6', alpha=0.7, markersize=6)  # 보라색으로 변경
            
        # Plot Retrain points (gray) starting from count1
        count2_display = min(count2, max_display)
        for y in range(count2_display):
            ax.plot(x, y + count1_display, 'o', color='gray', alpha=0.7, markersize=6)
        if count2 > max_display:
            ax.plot(x, 42, 'o', color='gray', alpha=0.7, markersize=6)
    
    # Calculate and find optimal threshold (최소 forgetting score를 찾음)
    scores = calculate_scores(values1, values2, bins, range_vals)
    best_score = min(scores, key=lambda x: x['forgetting_score'])
    
    # Add vertical line for optimal threshold
    ax.axvline(x=best_score['threshold'], color='red', linestyle='--', alpha=0.5)
    
    # Add text annotation
    text = f"Optimal threshold: {best_score['threshold']:.2f}\n"
    text += f"FPR: {best_score['fpr']:.3f}\n"
    text += f"FNR: {best_score['fnr']:.3f}\n"
    text += f"Epsilon: {best_score['epsilon']:.3f}\n"
    text += f"Forgetting score: {best_score['forgetting_score']:.3f}"
    
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            verticalalignment='top', 
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Add legend with file names
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label=legend2),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#8B5CF6', markersize=8, label=legend1)  # 보라색으로 변경
    ]
    ax.legend(handles=handles, loc='upper right')

def main():
    # Load distribution data and get file names for legend
    force_class = 9
    retrain_file = f'class_{force_class}_distribution_Retrain.json'
    ga_file = f'class_{force_class}_distribution_FT.json'
    
    # Extract legend names from file names (remove prefix and extension)
    legend_retrain = retrain_file.replace(f'class_{force_class}_distribution_', '').replace('.json', '')
    legend_ga = ga_file.replace(f'class_{force_class}_distribution_', '').replace('.json', '')
    
    with open(retrain_file, 'r') as f:
        retrain_data = json.load(f)
    with open(ga_file, 'r') as f:
        ga_data = json.load(f)

    plt.figure(figsize=(12, 5))

    # Left plot (Entropy)
    plt.subplot(1, 2, 1)
    ax1 = plt.gca()
    
    plot_distributions(ax1, 
                      ga_data['entropy']['values'],
                      retrain_data['entropy']['values'],
                      ga_data['entropy']['bins'],
                      ga_data['entropy']['range'],
                      legend_ga,
                      legend_retrain)

    make_break_marks(ax1, 0.9)
    plt.title(f'Class {force_class} Logit Entropy Distribution (Last 200)')
    plt.xlabel('Entropy')
    plt.ylabel('Count')
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(5))
    plt.xlim(0.0, 2.5)
    plt.ylim(-0.5, 43)
    plt.grid(True, alpha=0.2)

    # Right plot (Confidence)
    plt.subplot(1, 2, 2)
    ax2 = plt.gca()
    
    plot_distributions(ax2, 
                      ga_data['confidence']['values'],
                      retrain_data['confidence']['values'],
                      ga_data['confidence']['bins'],
                      ga_data['confidence']['range'],
                      legend_ga,
                      legend_retrain)

    make_break_marks(ax2, 0.9)
    plt.title(f'Class {force_class} Max Logit Confidence Distribution (Last 200)')
    plt.xlabel('Log Confidence Score')
    plt.ylabel('Count')
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(5))
    plt.xlim([-2.5, 10.0])
    plt.ylim(-0.5, 43)
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(f'class_{force_class}_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 