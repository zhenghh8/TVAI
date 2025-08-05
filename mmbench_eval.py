import argparse
import json
import os
import numpy as np

def extract_answer(ans_str):
    candidates = [c for c in ans_str if c in ('A', 'B', 'C', 'D')]
    return candidates[0] if candidates else None

def calculate_confusion_matrix(pred_list, label_list):
    class_names = ['A', 'B', 'C', 'D']
    class_index = {cls: i for i, cls in enumerate(class_names)}
    
    confusion_matrix = np.zeros((4, 4), dtype=int)
    
    for pred, label in zip(pred_list, label_list):
        if pred not in class_names or label not in class_names:
            continue
        
        true_idx = class_index[label]
        pred_idx = class_index[pred]
        
        confusion_matrix[true_idx][pred_idx] += 1
    
    return confusion_matrix, class_names

def save_acc(confusion_matrix, class_names, save_dir):
    metrics = {}
    n_classes = len(class_names)
    
    total_correct = np.trace(confusion_matrix)
    total_samples = np.sum(confusion_matrix)
    metrics['accuracy'] = total_correct / total_samples if total_samples > 0 else 0
    
    class_metrics = {}
    for i, cls in enumerate(class_names):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        tn = total_samples - (tp + fp + fn)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[cls] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    metrics['class_metrics'] = class_metrics
    
    precisions = [class_metrics[cls]['precision'] for cls in class_names]
    recalls = [class_metrics[cls]['recall'] for cls in class_names]
    f1s = [class_metrics[cls]['f1'] for cls in class_names]
    
    metrics['macro_precision'] = np.mean(precisions)
    metrics['macro_recall'] = np.mean(recalls)
    metrics['macro_f1'] = np.mean(f1s)


    with open(save_dir, 'w', encoding='utf-8') as f:
        f.write("Confusion matrix (row=label, col=pred)\n")
        f.write("\t" + "\t".join(class_names) + "\n")
        for i, row in enumerate(confusion_matrix):
            f.write(f"{class_names[i]}\t" + "\t".join(map(str, row)) + "\n")
        f.write("\n")
        
        f.write(f"Overall average accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro average percision: {metrics['macro_precision']:.4f}\n")
        f.write(f"Macro average recall: {metrics['macro_recall']:.4f}\n")
        f.write(f"Macro average F1: {metrics['macro_f1']:.4f}\n\n")
        
        f.write("Each options：\n")
        for cls in class_names:
            cm = metrics['class_metrics'][cls]
            f.write(f"Option {cls}：\n")
            f.write(f"  TP: {cm['TP']}, FP: {cm['FP']}, FN: {cm['FN']}, TN: {cm['TN']}\n")
            f.write(f"  Precision: {cm['precision']:.4f}, Recall: {cm['recall']:.4f}, F1: {cm['f1']:.4f}\n\n")

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    parser.add_argument("--ans_file", type=str, help="answer file")

    args = parser.parse_args()

    ans_path = 'results_generation/' + args.ans_file
    lines = open(ans_path).read().split("\n")

    pred_list = []
    label_list = []

    for line in lines:
        if len(line) == 0:
            break
        line = json.loads(line)
        pred_list.append(extract_answer(line["ans"]))
        label_list.append(line["label"])

    pos = 1
    neg = 0

    if not os.path.exists('results_analysis'):
        os.makedirs('results_analysis')
    base_name = os.path.splitext(os.path.basename(args.ans_file))[0]
    dir_name = os.path.dirname(args.ans_file)
    dir_name_1 = dir_name.split('/', 1)[0]
    dir_name_2 = dir_name.split('/', 2)[1]
    if not os.path.exists(os.path.join('results_analysis', dir_name_1)):
        os.makedirs(os.path.join('results_analysis', dir_name_1))
    if not os.path.exists(os.path.join('results_analysis', dir_name_1, dir_name_2)):
        os.makedirs(os.path.join('results_analysis', dir_name_1, dir_name_2))
    if not os.path.exists(os.path.join('results_analysis', dir_name)):
        os.makedirs(os.path.join('results_analysis', dir_name))
    
    save_acc_dir = os.path.join('results_analysis', dir_name, base_name + '_acc.txt')

    confusion_matrix, class_names = calculate_confusion_matrix(pred_list, label_list)
    save_acc(confusion_matrix, class_names, save_acc_dir)

