
import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def find_best_thresholds(predictions, true_labels_dict, thresholds):
    num_classes = len(predictions[0])
    best_thresholds = [0.5] * num_classes
    best_f1s = [0.0] * num_classes

    for class_idx in (range(num_classes)):
        for thresh in thresholds:
            f1 = f1_score(
                true_labels_dict[class_idx],
                predictions[thresh][class_idx],
                zero_division=0,
            )

            if f1 > best_f1s[class_idx]:
                best_f1s[class_idx] = f1
                best_thresholds[class_idx] = thresh
    
    return best_f1s, best_thresholds

def metrics_table(all_binary_results, all_true_labels):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    auc_scores = []

    num_classes = all_binary_results.shape[-1]
    for class_idx in range(num_classes):
        class_binary_results = all_binary_results[:, class_idx].numpy()
        class_true_labels = all_true_labels[:, class_idx].numpy()

        accuracy = accuracy_score(class_true_labels, class_binary_results)
        precision = precision_score(class_true_labels, class_binary_results, zero_division=0)
        recall = recall_score(class_true_labels, class_binary_results, zero_division=0)
        f1 = f1_score(class_true_labels, class_binary_results, zero_division=0)
        auc = roc_auc_score(class_true_labels, class_binary_results)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(auc)
    
    # no label
    class_binary_results = (~torch.sum(all_binary_results, axis = 1).bool()).int().numpy()
    class_true_labels = (~torch.sum(all_true_labels, axis = 1).bool()).int().numpy()

    accuracy = accuracy_score(class_true_labels, class_binary_results)
    precision = precision_score(class_true_labels, class_binary_results, zero_division=0)
    recall = recall_score(class_true_labels, class_binary_results, zero_division=0)
    f1 = f1_score(class_true_labels, class_binary_results, zero_division=0)
    auc = roc_auc_score(class_true_labels, class_binary_results)
    
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    auc_scores.append(auc)

    metrics_dict = {
        "Accuracy": accuracy_scores,
        # "Precision": precision_scores,
        # "Recall": recall_scores,
        "F1 Score": f1_scores,
        "AUC ROC": auc_scores,
    }

    return metrics_dict