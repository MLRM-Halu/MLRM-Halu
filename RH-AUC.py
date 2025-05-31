import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
import argparse
import  re

# RH-AUC  Visualization   
def plot_model_data(data, model_name, ax):

    data_sorted_x = sorted(data, key=lambda x: x[1])
    x = [point[1] for point in data_sorted_x]
    y = [point[0] for point in data_sorted_x]

    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    x_normalized = [(xi - x_min) / (x_max - x_min) for xi in x]
    y_normalized = [(yi - y_min) / (y_max - y_min) for yi in y]

    f_original = PchipInterpolator(x, y)
    x_smooth_original = np.linspace(min(x), max(x), 150)
    y_smooth_original = f_original(x_smooth_original)

    auc_smooth = np.trapz(y_normalized, x_normalized)
    print(f"AUC based on normalized data for {model_name}: {auc_smooth:.4f}")

    auc_original = np.trapz(y, x) / (max(x) - min(x))
    print(f"AUC based on original data for {model_name}: {auc_original:.4f}")

    ax.plot(x_smooth_original, y_smooth_original, label=f'{model_name} Smooth Curve', linestyle='-', linewidth=2)
    ax.plot(x, y, marker='o', linestyle='', markersize=8, label=f'{model_name} Data Points')

    ax.fill_between(x_smooth_original, y_smooth_original, alpha=0.3)

    ax.set_xlabel('Reason Score', fontsize=10)
    ax.set_ylabel('Perception Score', fontsize=10)
    ax.set_title(f'{model_name} (AUC = {auc_smooth:.2f})', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    y_margin = (max(y) - min(y)) * 0.12
    ax.set_ylim(min(y) - y_margin, max(y) + y_margin)

#   You can also adjust it according to your format
def extract_scores_from_txt(file_path):
    scores = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                model_name = re.search(r"File:\s*(.*?)\.json", line).group(1)  # Extract model name from file path
                accuracy = re.search(r"Accuracy:\s*([\d\.]+)%", line).group(1)  # Extract accuracy percentage
                correct_count = int(re.search(r"\((\d+)/", line).group(1))  # Extract correct answers count
                total_count = int(re.search(r"/(\d+)\)", line).group(1))  # Extract total answers count

                scores.append((model_name, accuracy, correct_count, total_count))
    except Exception as e:
        print(f"Error extracting scores from {file_path}: {str(e)}")
    return scores


def plot_combined_results(txt_file_reason, txt_file_hallu):

    model_results_1 = extract_scores_from_txt(txt_file_reason)
    model_results_2 = extract_scores_from_txt(txt_file_hallu)

    combined_results = model_results_1 + model_results_2

    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    axs = axs.flatten()

    for i, (model_name, accuracy, correct_count, total_count) in enumerate(combined_results):
        if i >= len(axs):
            break
        plot_model_data([(float(accuracy), float(accuracy))], model_name, axs[i])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process and visualize model evaluation results")
    parser.add_argument('--txt_file_reason', type=str, default='/path/to/default/evaluation_summary_reason.txt',
                        help="Path to the first TXT file")
    parser.add_argument('--txt_file_hallu', type=str, default='/path/to/default/evaluation_summary_hallucination.txt',
                        help="Path to the second TXT file")

    args = parser.parse_args()
    plot_combined_results(args.txt_file_reason, args.txt_file_hallu)