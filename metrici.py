import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import *

def read_data_gt(file_path):
    annotations = []

    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            x1 = x_center - width/2
            y1 = y_center - height/2

            x2 = x_center + width/2
            y2 = y_center + height/2

            bbox_pixels = [x1, y1, x2, y2]
            annotations.append((class_id, bbox_pixels))

    return annotations

def read_data_pred(file_path):
    annotations = []
    img_height = 640
    img_width = 640

    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            for line in file:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                confidence = float(parts[5])

                x1 = (x_center - width / 2.)/img_width
                y1 = (y_center - height / 2.)/img_height

                x2 = (x_center + width / 2.)/img_width
                y2 = (y_center + height / 2.)/img_height

                bbox_pixels = [x1, y1, x2, y2]
                annotations.append((class_id, bbox_pixels, confidence))
    else:
        annotations = []
    return annotations


def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area_box1 + area_box2 - intersection

    if union == 0:
        return 0
    return intersection / union


def build_confusion_matrix(gt_dir, pred_dir, iou_threshold=0.5):

    y_true = []
    y_pred = []
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith(".txt")]

    for gt_file in gt_files:
        gt_path = os.path.join(gt_dir, gt_file)
        pred_path = os.path.join(pred_dir, gt_file)

        ground_truth = read_data_gt(gt_path)
        predictions = read_data_pred(pred_path)
        if not predictions:
            predictions = [-1] * len(ground_truth)
            continue
        matched_gt = set()

        for pred_class, pred_bbox, confidence in predictions:
            matched = False
            for gt_index, (gt_class, gt_bbox) in enumerate(ground_truth):
                if gt_index in matched_gt:
                    continue
                iou = calculate_iou(gt_bbox, pred_bbox)
                if iou >= iou_threshold:
                    y_true.append(gt_class)
                    y_pred.append(pred_class)
                    matched_gt.add(gt_index)
                    matched = True
                    break

            if not matched:
                y_pred.append(pred_class)
                y_true.append(-1)

        # False negative
        for gt_index, (gt_class, gt_bbox) in enumerate(ground_truth):
            if gt_index not in matched_gt:
                y_true.append(gt_class)
                y_pred.append(-1)  #Clasă nedetectată

    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, class_labels):
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_labels))) + [-1]).astype('float')

    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix = conf_matrix / row_sums

    # Extindem etichetele claselor pentru a include FN și FP
    extended_labels = class_labels + ["Unknown"]

    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matricea de confuzie")
    plt.colorbar()

    tick_marks = np.arange(len(extended_labels))
    plt.xticks(tick_marks, extended_labels, rotation=45)
    plt.yticks(tick_marks, extended_labels)

    thresh = conf_matrix.max() / 2.
    for i, j in np.ndindex(conf_matrix.shape):
        plt.text(j, i, f"{conf_matrix[i, j]:.2f}", horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel('Ground truth')
    plt.xlabel('Predictii')

    precision = precision_score(y_true, y_pred, average=None, labels=range(len(class_labels)))
    recall = recall_score(y_true, y_pred, average=None, labels=range(len(class_labels)))
    f1 = f1_score(y_true, y_pred, average=None, labels=range(len(class_labels)))

    report = ""
    for i, class_name in enumerate(class_labels):
        report += f"{class_name}: Precision: {precision[i]:.2f}, Recall: {recall[i]:.2f}, F1 Score: {f1[i]:.2f}\n"

    plt.gcf().text(0.05, 0.95, report, fontsize=10, va='top', ha='left', bbox=dict(facecolor='white', alpha=1))

    plt.show()


if __name__ == '__main__':
    gt_dir = "res/images/test/labels"
    pred_dir = "res/images/test/_labels"
    class_labels = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
    y_true, y_pred = build_confusion_matrix(gt_dir, pred_dir, iou_threshold=0.1)
    plot_confusion_matrix(y_true, y_pred, class_labels)