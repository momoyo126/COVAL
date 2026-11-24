"""
Evaluate STU anomaly detection from saved anomaly scores or full class outputs.

    sequences_anomscore/{sequence}/{frame}.npy -> metric json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN
from sklearn.metrics import auc, average_precision_score, roc_curve
from tqdm import tqdm


def load_point_cloud(scan_path):
    scan = np.fromfile(scan_path, dtype=np.float32).reshape((-1, 4))
    return scan[:, :3], scan[:, 3:]


def load_labels(label_path):
    labels = np.fromfile(label_path, dtype=np.uint32).astype(np.int32)
    semantic_label = labels & 0xFFFF
    instance_label = labels >> 16
    return semantic_label, instance_label


def convert_to_builtin_types(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class UQEvaluator:
    def __init__(self, min_points=5, num_classes=2, ignore_labels=None, instance_labels=None):
        self.min_points = min_points
        self.num_classes = num_classes
        self.ignore_labels = [255] if ignore_labels is None else ignore_labels
        self.instance_labels = [1] if instance_labels is None else instance_labels
        self.reset()

    def reset(self):
        self.pan_tp_uq = np.zeros(self.num_classes, dtype=np.int64)
        self.pan_iou_uq = np.zeros(self.num_classes, dtype=np.double)
        self.pan_fn_uq = np.zeros(self.num_classes, dtype=np.int64)
        self.pan_fp_uq = np.zeros(self.num_classes, dtype=np.int64)

    def add_batch_for_uq(self, points, prediction, semantic_label, instance_label):
        prediction = np.asarray(prediction)
        instance_prediction = np.zeros(prediction.shape, dtype=np.int64)
        if len(points[prediction]) > 0:
            clusters = DBSCAN(eps=1.0, min_samples=1, n_jobs=-1).fit(points[prediction]).labels_
            instance_prediction[prediction] = clusters + 1

        self.add_batch_unknown(
            prediction,
            instance_prediction,
            semantic_label,
            instance_label,
        )

    def add_batch_unknown(self, x_sem_row, x_inst_row, y_sem_row, y_inst_row):
        x_inst_row = x_inst_row + 1
        y_inst_row = y_inst_row + 1

        for cl in self.ignore_labels:
            gt_not_in_excl_mask = y_sem_row != cl
            x_sem_row = x_sem_row[gt_not_in_excl_mask]
            y_sem_row = y_sem_row[gt_not_in_excl_mask]
            x_inst_row = x_inst_row[gt_not_in_excl_mask]
            y_inst_row = y_inst_row[gt_not_in_excl_mask]

        for cl in self.instance_labels:
            x_inst_in_cl_mask = x_sem_row == cl
            y_inst_in_cl_mask = y_sem_row == cl
            x_inst_in_cl = x_inst_row * x_inst_in_cl_mask.astype(np.int64)
            y_inst_in_cl = y_inst_row * y_inst_in_cl_mask.astype(np.int64)

            unique_pred, counts_pred = np.unique(
                x_inst_in_cl[x_inst_in_cl > 0], return_counts=True
            )
            id2idx_pred = {id_: idx for idx, id_ in enumerate(unique_pred)}
            matched_pred = np.array([False] * unique_pred.shape[0])

            unique_gt, counts_gt = np.unique(
                y_inst_in_cl[y_inst_in_cl > 0], return_counts=True
            )
            id2idx_gt = {id_: idx for idx, id_ in enumerate(unique_gt)}
            matched_gt = np.array([False] * unique_gt.shape[0])

            valid_combos = np.logical_and(x_inst_in_cl > 0, y_inst_in_cl > 0)
            offset_combo = x_inst_in_cl[valid_combos] + 2**32 * y_inst_in_cl[valid_combos]
            unique_combo, counts_combo = np.unique(offset_combo, return_counts=True)

            gt_labels = unique_combo // 2**32
            pred_labels = unique_combo % 2**32
            gt_areas = np.array([counts_gt[id2idx_gt[id_]] for id_ in gt_labels])
            pred_areas = np.array([counts_pred[id2idx_pred[id_]] for id_ in pred_labels])
            intersections = counts_combo
            unions = gt_areas + pred_areas - intersections
            ious = intersections.astype(np.float64) / unions.astype(np.float64)

            tp_indexes = ious > 0.5
            self.pan_tp_uq[cl] += np.sum(tp_indexes)
            self.pan_iou_uq[cl] += np.sum(ious[tp_indexes])

            matched_gt[[id2idx_gt[id_] for id_ in gt_labels[tp_indexes]]] = True
            matched_pred[[id2idx_pred[id_] for id_ in pred_labels[tp_indexes]]] = True

            self.pan_fn_uq[cl] += np.sum(
                np.logical_and(counts_gt >= self.min_points, matched_gt == False)
            )
            self.pan_fp_uq[cl] += np.sum(
                np.logical_and(counts_pred >= self.min_points, matched_pred == False)
            )

    def get_stats(self):
        return (
            self.pan_iou_uq[1],
            self.pan_tp_uq[1],
            self.pan_fp_uq[1],
            self.pan_fn_uq[1],
        )

    def get_uq(self):
        sq_uq_all = self.pan_iou_uq.astype(np.float64) / np.maximum(
            self.pan_tp_uq.astype(np.float64), 1e-15
        )
        recallq_all = self.pan_tp_uq.astype(np.float64) / np.maximum(
            self.pan_tp_uq.astype(np.float64) + self.pan_fn_uq.astype(np.float64), 1e-15
        )
        uq_all = sq_uq_all * recallq_all
        return sq_uq_all, recallq_all, uq_all


def calculate_auroc(predictions, targets):
    fpr, tpr, thresholds = roc_curve(y_true=targets, y_score=predictions)
    roc_auc = auc(fpr, tpr)
    fpr_best = 0
    threshold = 0
    for tpr_val, fpr_val, threshold in zip(tpr, fpr, thresholds):
        if tpr_val > 0.95:
            fpr_best = fpr_val
            break
    return roc_auc, fpr_best, threshold


def calculate_ood_metrics(targets, predictions):
    if not targets or sum(len(value) for value in targets) == 0:
        return dict(AP=None, FPR95=None, AUROC=None, threshold=None)
    targets = np.concatenate(targets, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    if len(np.unique(targets)) < 2:
        return dict(AP=None, FPR95=None, AUROC=None, threshold=None)
    ap = average_precision_score(y_true=targets, y_score=predictions)
    roc_auc, fpr, threshold = calculate_auroc(predictions=predictions, targets=targets)
    return {
        "AP": ap * 100,
        "FPR95": fpr * 100,
        "AUROC": roc_auc * 100,
        "threshold": threshold,
    }


def softmax_np(logits):
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def scores_from_conf(conf_path, prediction_type, input_type="probabilities"):
    conf = np.load(conf_path).astype(np.float32)
    if conf.ndim == 1:
        if input_type != "probabilities" or prediction_type != "anomlogit":
            raise ValueError("Scalar anomaly scores require probability input and anomlogit prediction type")
        if not np.isfinite(conf).all() or (conf < 0).any() or (conf > 1).any():
            raise ValueError(f"Expected finite anomaly probabilities in [0, 1]: {conf_path}")
        return conf.astype(np.float16)
    if conf.ndim != 2 or conf.shape[1] != 20 or not np.isfinite(conf).all():
        raise ValueError(f"Expected finite (N, 20) classifier outputs: {conf_path}")
    if input_type == "logits":
        conf = softmax_np(conf)
    elif input_type == "probabilities":
        if (conf < 0).any() or (conf > 1).any() or not np.allclose(conf.sum(1), 1, atol=0.003):
            raise ValueError(f"Input is not a probability distribution: {conf_path}")
    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    if prediction_type == "anomlogit":
        return conf[:, 19].astype(np.float16)
    if prediction_type == "maxlogit":
        max_logit = (np.max(conf[:, :19], axis=1) * -1) + 1
        return max_logit.astype(np.float16)

    raise ValueError(f"Unsupported prediction_type: {prediction_type}")


def process_sequence(sequence, data_path, conf_path, prediction_type, input_type="probabilities"):
    pc_indices = sorted(
        int(f.stem) for f in (data_path / sequence / "velodyne").glob("*.bin")
    )

    sequence_data = {
        "sequence_name": sequence,
        "anomaly_sizes": [],
        "unique_labels": [],
        "no_labels_frames": [],
        "uq": [],
        "metrics": None,
    }

    predictions_list = []
    labels_list = []
    unique_list = []
    no_labels = []
    anomaly_size = []
    distances_list = []

    uq_per_sequence = UQEvaluator()

    for index in pc_indices:
        points, _ = load_point_cloud(data_path / sequence / "velodyne" / f"{index:06}.bin")
        semantic_labels, instance_label = load_labels(
            data_path / sequence / "labels" / f"{index:06}.label"
        )
        prediction = scores_from_conf(
            conf_path / sequence / f"{index:06}.npy",
            prediction_type,
            input_type,
        )
        if len(prediction) != len(points):
            raise ValueError(f"Point/score count mismatch in sequence {sequence}, frame {index:06}")

        distances = np.linalg.norm(points, axis=1)
        unique_labels = np.unique(semantic_labels)
        inlier_labels = np.where(semantic_labels != 0, 0, -1)
        processed_labels = np.where(semantic_labels == 2, 1, inlier_labels)
        processed_labels = np.where(
            (distances > 50) | (distances < 2.5),
            -1,
            processed_labels,
        )
        labels = processed_labels[processed_labels != -1]

        if sum(labels) < 5:
            no_labels.append(index)
            continue

        valid_mask = processed_labels != -1
        uq_per_sequence.add_batch_for_uq(
            points[valid_mask],
            prediction[valid_mask] > 0.5,
            labels,
            instance_label[valid_mask],
        )

        prediction = prediction[valid_mask]
        distances = distances[valid_mask]

        distances_list.append(distances)
        labels_list.append(labels)
        predictions_list.append(prediction)
        unique_list.append(unique_labels)
        anomaly_size.append(np.sum(labels == 1))

    sq, rq, uq = uq_per_sequence.get_uq()
    print(f"seq: {sequence} sq: {sq[1] * 100} rq: {rq[1] * 100} uq: {uq[1] * 100}")

    sequence_data.update(
        {
            "metrics": calculate_ood_metrics(labels_list, predictions_list),
            "anomaly_sizes": anomaly_size,
            "unique_labels": list(np.unique(np.concatenate(unique_list))) if unique_list else [],
            "no_labels_frames": no_labels,
            "uq": uq_per_sequence.get_stats(),
        }
    )
    print(sequence_data["metrics"])

    return labels_list, predictions_list, distances_list, sequence_data


def compute_metrics(data_path, conf_path, output_path, prediction_type, num_jobs, input_type="probabilities"):
    overall_metrics = {"sequence_metrics": [], "score_input_type": input_type,
                       "prediction_type": prediction_type}
    overall_predictions = []
    overall_labels = []
    overall_distances = []

    sequence_paths = sorted(list(data_path.glob("1*")))
    print(f"sequences: {len(sequence_paths)}")
    print(f"sequences: {[p.stem for p in sequence_paths]}")

    results = Parallel(n_jobs=num_jobs)(
        delayed(process_sequence)(sequence_path.name, data_path, conf_path, prediction_type, input_type)
        for sequence_path in tqdm(sequence_paths, leave=False)
    )

    for labels_list, predictions_list, distances_list, sequence_data in results:
        overall_metrics["sequence_metrics"].append(sequence_data)
        overall_labels.extend(labels_list)
        overall_predictions.extend(predictions_list)
        overall_distances.extend(distances_list)

    pan_iou_uq = []
    pan_tp_uq = []
    pan_fp_uq = []
    pan_fn_uq = []
    for stats in overall_metrics["sequence_metrics"]:
        pan_iou_uq.append(stats["uq"][0])
        pan_tp_uq.append(stats["uq"][1])
        pan_fp_uq.append(stats["uq"][2])
        pan_fn_uq.append(stats["uq"][3])

    pan_iou_uq = sum(pan_iou_uq)
    pan_tp_uq = sum(pan_tp_uq)
    pan_fp_uq = sum(pan_fp_uq)
    pan_fn_uq = sum(pan_fn_uq)

    sq_uq_all = pan_iou_uq.astype(np.float64) / np.maximum(
        pan_tp_uq.astype(np.float64), 1e-15
    )
    recallq_all = pan_tp_uq.astype(np.float64) / np.maximum(
        pan_tp_uq.astype(np.float64) + pan_fn_uq.astype(np.float64), 1e-15
    )
    uq_all = sq_uq_all * recallq_all

    rq_all = pan_tp_uq.astype(np.float64) / np.maximum(
        pan_tp_uq.astype(np.float64)
        + 0.5 * pan_fp_uq.astype(np.float64)
        + 0.5 * pan_fn_uq.astype(np.float64),
        1e-15,
    )
    pq_all = sq_uq_all * rq_all

    overall_metrics["overall_metrics"] = {
        "semantic": calculate_ood_metrics(overall_labels, overall_predictions),
        "object": {
            "SQ": sq_uq_all * 100,
            "RecallQ": recallq_all * 100,
            "UQ": uq_all * 100,
            "RQ": rq_all * 100,
            "PQ": pq_all * 100,
        },
    }
    print(overall_metrics["overall_metrics"])

    distance_thresholds = [0, 10, 20, 30, 40, 50]
    overall_metrics["distance_threshold_metrics"] = {}

    for threshold_idx in range(1, len(distance_thresholds)):
        threshold_labels = []
        threshold_predictions = []

        for labels, predictions, distances in zip(
            overall_labels,
            overall_predictions,
            overall_distances,
        ):
            mask_within_threshold = (
                distances > distance_thresholds[threshold_idx - 1]
            ) & (distances <= distance_thresholds[threshold_idx])
            threshold_labels.append(labels[mask_within_threshold])
            threshold_predictions.append(predictions[mask_within_threshold])

        overall_metrics["distance_threshold_metrics"][f"{distance_thresholds[threshold_idx]}m"] = {
            "metric": calculate_ood_metrics(threshold_labels, threshold_predictions),
        }

    print(overall_metrics["overall_metrics"])
    print(overall_metrics["distance_threshold_metrics"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(overall_metrics, f, indent=4, default=convert_to_builtin_types)
    print(f"Metrics logged to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute STU anomaly metrics from saved anomaly scores or class outputs."
    )
    parser.add_argument("--data-path", type=Path, default=Path("../dataset/STU/validation"))
    parser.add_argument("--conf-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--prediction-type",
        choices=["anomlogit", "maxlogit"],
        default="anomlogit",
    )
    parser.add_argument("--num-jobs", type=int, default=4)
    parser.add_argument("--input-type", choices=["probabilities", "logits"], default="probabilities",
                        help="The released tester saves probabilities; do not softmax these again")
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)
    compute_metrics(
        data_path=args.data_path,
        conf_path=args.conf_path,
        output_path=args.output_path,
        prediction_type=args.prediction_type,
        num_jobs=args.num_jobs,
        input_type=args.input_type,
    )


if __name__ == "__main__":
    main()
