import pandas as pd
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


def get_categories(tree):
    node_state = np.asarray(tree.node_state)
    col1 = node_state[:, 0]
    col2 = node_state[:, 1]

    conditions = [
        (col1 == True)  & (col2 == False), # Missing
        (col1 == False) & (col2 == True),  # Known
        (col1 == False) & (col2 == False)  # Unknown
    ]
    choices = ["missing", "known", "unknown"]
    categories = np.select(conditions, choices, default="known")
    categories = np.append(categories, "none")

    return categories

def get_predictions(predictions_path, class_level):
    
    modelResults_df = pd.read_csv(predictions_path)
    modelResults_df[f"{class_level}_prob"] = modelResults_df[f"{class_level}_prob"].fillna(0.0)
    predictions_list = modelResults_df[f"{class_level}_id"].tolist()
    probs_list = modelResults_df[f"{class_level}_prob"].tolist()

    return predictions_list, probs_list



def plot_cumulative(list_of_results, accuracies, config, exp_details, class_level, id, test_mode):
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
    plt.savefig(f"{id}_{test_mode}_{class_level}.png")
    plt.close()

    return

def get_confusion_matrix(pred_cats, label_cats, class_level, id):
  categories = ["known", "missing", "unknown"]
  cm = confusion_matrix(label_cats, pred_cats, labels=categories)
  # plt.figure(figsize=(8, 6))
  # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
  #             xticklabels=categories, yticklabels=categories)
  
  # plt.ylabel('Actual Category')
  # plt.xlabel('Predicted Category')
  # plt.title(f'Confusion Matrix: {class_level}')
  
  # plt.tight_layout()
  # plt.savefig(f"{id}_{class_level}_cm.png")
  # plt.close()
  
  print("Rows = Actual, Columns = Predicted\n")

  print(categories)
  print(cm)

def calculate_correctness(predictions_list, probs_list, labels_list, node_categories, class_level, id):
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

  # Calculate category accuracies on aligned arrays only
  pred_cats = node_categories[preds_valid]
  label_cats = node_categories[labels_valid]
  # category_accuracies = {}
  # for cat in ["known", "missing", "unknown"]:
  #     mask = (label_cats == cat)
  #     if np.any(mask):
  #         category_accuracies[cat] = np.mean(correctness[mask]) * 100
  #     else:
  #         category_accuracies[cat] = np.nan

  # get_confusion_matrix(pred_cats, label_cats, class_level, id)   # Creating Confusion Matrix

  print(f"Accuracy: {accuracy:.2f}%")
  # print("Category accuracies: \n", category_accuracies)

  return results, accuracy


def evaluate(predictions_path, labels_path, train_config, exp_details, class_level, tax_dir, id):
  predictions_list, probs_list = get_predictions(predictions_path, class_level)  
  labels_df = pd.read_csv(labels_path)
  
  test_mode = str(labels_path).split("/")[-1].split("_")[0]
  labels_list = labels_df[f"{class_level}_id"].tolist()
  validate_lengths(predictions_list, labels_list)

  tree, _, _, _ = protax_utils.read_model_jax("models/scalings/plain.npz", tax_dir)
  node_categories = get_categories(tree)

  results, accuracy = calculate_correctness(predictions_list, probs_list, labels_list, node_categories, class_level, id)
  plot_cumulative([results], [accuracy], train_config, exp_details, class_level, id, test_mode)

  return