# Imports
import logging
import pathlib
from itertools import chain
import sys
import json

from pytorch_faster_rcnn_tutorial.metrics.enumerators import MethodAveragePrecision
from pytorch_faster_rcnn_tutorial.metrics.pascal_voc_evaluator import (
    get_pascalvoc_metrics,
)
from pytorch_faster_rcnn_tutorial.utils import (
    from_file_to_boundingbox,
    get_filenames_of_path,
)

logger: logging.Logger = logging.getLogger(__name__)

# root directory
FOLDER_PATH = pathlib.Path.cwd()
MODULE_PATH = FOLDER_PATH / "pytorch_faster_rcnn_tutorial"
DATA_PATH = MODULE_PATH / "data" / "heads"


def main():
    # input and target files
    predictions = get_filenames_of_path(DATA_PATH / "predictions")
    predictions.sort()

    # get the gt_boxes from disk
    gt_boxes = [
        from_file_to_boundingbox(file_name, groundtruth=True)
        for file_name in predictions
    ]
    # reduce list
    gt_boxes = list(chain(*gt_boxes))

    pred_boxes = [
        from_file_to_boundingbox(file_name, groundtruth=False)
        for file_name in predictions
    ]
    pred_boxes = list(chain(*pred_boxes))

    output = get_pascalvoc_metrics(
        gt_boxes=gt_boxes,
        det_boxes=pred_boxes,
        iou_threshold=0.5,
        method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
        generate_table=True,
    )

    per_class, m_ap = output["per_class"], output["m_ap"]
    head = per_class[1]
    
    with open("metrics.json", "w") as f:
        json.dump({
            'precision': head["precision"].tolist(),
            'recall': head["recall"].tolist(),
            'AP': float(head["AP"]),
            'interpolated precision': head["interpolated precision"],
            'interpolated recall': head["interpolated recall"],
            'total positives': int(head["total positives"]),
            'total tp': int(head["total tp"]),
            'total fp': int(head["total fp"]),
            'method': str(head["method"]),
            'iou': float(head["iou"]),
        }, f, indent=4)

if __name__ == "__main__":
    main()
