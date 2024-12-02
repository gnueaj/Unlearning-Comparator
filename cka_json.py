import json
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_cka_matrix(ax, matrix, title, layers):
    im = ax.imshow(matrix, cmap='magma', interpolation='none', origin='upper')
    ax.set_title(title, fontsize=28)
    ax.set_xlabel('After Unlearning', fontsize=24)
    ax.set_ylabel('Before Unlearning', fontsize=24)
    ax.set_xticks(range(len(matrix)))
    ax.set_yticks(range(len(matrix)))
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=16)
    ax.set_yticklabels(layers, fontsize=16)

    # Add text annotations with larger font
    # for i in range(len(matrix)):
    #     for j in range(len(matrix)):
    #         ax.text(j, i, f"{matrix[i][j]:.3f}", 
    #                 ha="center", va="center", color="w", fontsize=12)

    return im

def visualize_cka_similarity(baseline_json_path, comparison_json_path, output_path):
    # Load JSON data for both models
    with open(baseline_json_path, 'r') as file:
        baseline_data = json.load(file)
    with open(comparison_json_path, 'r') as file:
        comparison_data = json.load(file)

    # Extract similarity data
    baseline_similarity = baseline_data['cka']
    comparison_similarity = comparison_data['cka']
    layers = baseline_similarity['layers']

    # Create subplots for heatmaps and line plot
    fig = plt.figure(figsize=(24, 30))
    gs = fig.add_gridspec(3, 2, height_ratios=[3.5, 4, 4], width_ratios=[1, 1])

    ax5 = fig.add_subplot(gs[0, :])  # Line plot at the top
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])

    # Plot baseline train data
    im1 = plot_cka_matrix(ax1, baseline_similarity['train']['forget_class'], 
                    "Baseline (Forget Class)", layers)
    im3 = plot_cka_matrix(ax2, comparison_similarity['train']['forget_class'], 
                    "Comparison (Forget Class)", layers)

    # Plot comparison train data
    im2 = plot_cka_matrix(ax3, baseline_similarity['train']['other_classes'], 
                    "Baseline (Remain Classes)", layers)
    im4 = plot_cka_matrix(ax4, comparison_similarity['train']['other_classes'], 
                    "Comparison (Remain Classes)", layers)

    # Add colorbars in the empty space of each row
    # 2행 컬러바
    cbar_ax2 = ax2.inset_axes([1.05, 0.15, 0.05, 0.7])
    fig.colorbar(im2, cax=cbar_ax2, label='CKA Similarity').ax.tick_params(labelsize=24)
    
    # 3행 컬러바
    cbar_ax3 = ax4.inset_axes([1.05, 0.15, 0.05, 0.7])
    fig.colorbar(im4, cax=cbar_ax3, label='CKA Similarity').ax.tick_params(labelsize=24)

    # Adjust subplot spacing
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    # Add line plot of diagonal values
    x = np.arange(len(layers))
    
    # Baseline diagonals
    diag_baseline_forget = np.diag(baseline_similarity['train']['forget_class'])
    diag_baseline_other = np.diag(baseline_similarity['train']['other_classes'])
    
    # Comparison diagonals
    diag_comparison_forget = np.diag(comparison_similarity['train']['forget_class'])
    diag_comparison_other = np.diag(comparison_similarity['train']['other_classes'])

    # Plot with different markers and colors
    ax5.plot(x, diag_baseline_forget, '-', color='red', marker='o', 
             label='Baseline (Forget Class)', markersize=24, linewidth=2)
    ax5.plot(x, diag_baseline_other, '-', color='green', marker='o', 
             label='Baseline (Remain Classes)', markersize=24, linewidth=2)
    ax5.plot(x, diag_comparison_forget, '--', color='red', marker='^', 
             label='Comparison (Forget Class)', markersize=24, linewidth=2)
    ax5.plot(x, diag_comparison_other, '--', color='green', marker='^', 
             label='Comparison (Remain Classes)', markersize=24, linewidth=2)

    ax5.set_ylabel('CKA Similarity', fontsize=24)
    ax5.set_title('Layer-wise CKA Similarity (Before vs. After Unlearning)', fontsize=36)
    ax5.set_xticks(range(len(layers)))
    ax5.set_xticklabels(layers, rotation=45, ha='right', fontsize=20)
    ax5.grid(True, linestyle='--', alpha=0.7)
    ax5.legend(fontsize=24, markerscale=1.0)

    # Replace tight_layout with manual adjustment
    plt.subplots_adjust(right=0.85)  # Make room for colorbars
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python cka_json.py <BASELINE_FILE_NAME> <COMPARISON_FILE_NAME>")
        sys.exit(1)
        
    BASELINE_FILE = sys.argv[1]
    COMPARISON_FILE = sys.argv[2]
    BASELINE_JSON_PATH = f'/home/jaeung/mu-dashboard/backend/data/{BASELINE_FILE}.json'
    COMPARISON_JSON_PATH = f'/home/jaeung/mu-dashboard/backend/data/{COMPARISON_FILE}.json'
    output_path = f'cka_comp.png'
    visualize_cka_similarity(BASELINE_JSON_PATH, COMPARISON_JSON_PATH, output_path)