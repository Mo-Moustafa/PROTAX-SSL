import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import textwrap
from protax import protax_utils
import seaborn as sns
from sklearn.metrics import confusion_matrix

def validate_lengths(predictions_list, labels_list):
    if len(predictions_list) != len(labels_list):
        raise ValueError("Predictions and labels lists must have the same length."
                         f" Got {len(predictions_list)} and {len(labels_list)}.")

    return


def get_predictions(predictions_path, class_level):
    
    modelResults_df = pd.read_csv(predictions_path)
    modelResults_df[f"{class_level}_prob"] = modelResults_df[f"{class_level}_prob"].fillna(0.0)
    predictions_list = modelResults_df[f"{class_level}_id"].tolist()
    probs_list = modelResults_df[f"{class_level}_prob"].tolist()

    return predictions_list, probs_list


def plot_cumulative(list_of_results, accuracies, config, exp_details, class_level, id, test_mode, loo_id):
    wrapper = textwrap.TextWrapper(width=70)
    config = "\n".join(wrapper.wrap(text=config))

    plt.figure(figsize=(8, 6))
    plt.plot([0, 100], [0, 100],  label="Ideal Calibration", color='gray')

    for results in list_of_results:
      # Sort by predicted probabilities (ascending order)
      sorted_indices = np.argsort(results['probs'])
      sorted_probs = results['probs'][sorted_indices]
      sorted_correct = results['correctness'][sorted_indices]

      # Compute cumulative probability sum
      cumulative_probs = np.cumsum(sorted_probs)  # Raw cumulative probability sums
      cumulative_correct = np.cumsum(sorted_correct)  # Raw cumulative correct sums

      n = len(sorted_correct)
      cumulative_probs =  cumulative_probs / n * 100  # Normalize to percentage
      # cumulative_probs = (cumulative_probs / cumulative_probs[-1]) * 100
      cumulative_correct = cumulative_correct / n * 100  # Normalize to percentage

      # Plot cumulative probability vs. cumulative correct
      plt.plot(cumulative_probs, cumulative_correct, color='black')

      # Highlight the last point with a marker
      plt.plot(cumulative_probs[-1], cumulative_correct[-1], marker='o', markersize=8, color='black')

    plt.xlabel("Cumulative Probability")
    plt.ylabel("Cumulative Correct")
    plt.title(f"{config}\n {exp_details}\n {class_level} accuracy={np.mean(accuracies)}%")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if loo_id is not None:
        plt.savefig(f"{id}_{test_mode}_{class_level}_{loo_id}.png")
    else:
        plt.savefig(f"{id}_{test_mode}_{class_level}.png")    
    plt.close()

    return


def calculate_correctness(predictions_list, probs_list, labels_list):
  preds = np.asarray(predictions_list)
  probs = np.asarray(probs_list)
  labels = np.asarray(labels_list)

  valid_mask = labels != -1
  preds_valid = preds[valid_mask]
  labels_valid = labels[valid_mask]
  probs_valid = probs[valid_mask]

  correctness = (preds_valid == labels_valid).astype(int)
  results = {
          "probs": probs_valid,
          "correctness": correctness
  }

  accuracy = float(correctness.mean() * 100)
  print(f"Accuracy: {accuracy:.2f}%")

  return results, accuracy


def evaluate(predictions_path, labels_path, train_config, exp_details, class_level, tax_dir, id, loo_id=None):
  predictions_list, probs_list = get_predictions(predictions_path, class_level)  
  labels_df = pd.read_csv(labels_path)
  
  test_mode = str(labels_path).split("/")[-1].split("_")[0]
  labels_list = labels_df[f"{class_level}_id"].tolist()
  validate_lengths(predictions_list, labels_list)

  results, accuracy = calculate_correctness(predictions_list, probs_list, labels_list)
  plot_cumulative([results], [accuracy], train_config, exp_details, class_level, id, test_mode, loo_id=loo_id)

  return