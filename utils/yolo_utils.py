import torch
from torch import masked_select

def yolo_filter_boxes(box_confidence,
                      boxes,
                      box_class_probs,
                      threshold=.6):
    box_scores = box_confidence * box_class_probs
    class_scores, classes = torch.max(box_scores, 1)

    threshold_mask = class_scores >= threshold

    scores = masked_select(class_scores, threshold_mask)
    boxes = masked_select(boxes, threshold_mask)
    classes = masked_select(classes, threshold_mask)

    return scores, boxes, classes
