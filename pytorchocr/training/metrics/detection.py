import numpy as np
import torch
from shapely.geometry import Polygon


class DetectionIoUEvaluator:
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
        self.iou_constraint = float(iou_constraint)
        self.area_precision_constraint = float(area_precision_constraint)

    @staticmethod
    def _polygon(points):
        polygon = Polygon(np.asarray(points, dtype=np.float32))
        if not polygon.is_valid or polygon.area <= 0:
            return None
        return polygon

    def evaluate_image(self, ground_truth, predictions):
        gt_polygons = []
        gt_ignored = []
        for item in ground_truth:
            polygon = self._polygon(item["points"])
            if polygon is not None:
                gt_polygons.append(polygon)
                gt_ignored.append(bool(item.get("ignore", False)))

        det_polygons = []
        for item in predictions:
            polygon = self._polygon(item["points"])
            if polygon is not None:
                det_polygons.append(polygon)

        det_ignored = set()
        ignored_gt_indices = {
            index for index, ignored in enumerate(gt_ignored) if ignored
        }
        for det_index, det_polygon in enumerate(det_polygons):
            for gt_index in ignored_gt_indices:
                overlap = gt_polygons[gt_index].intersection(det_polygon).area
                if overlap / det_polygon.area > self.area_precision_constraint:
                    det_ignored.add(det_index)
                    break

        matched_gt = set()
        matched_det = set()
        matched = 0
        for gt_index, gt_polygon in enumerate(gt_polygons):
            if gt_index in ignored_gt_indices:
                continue
            for det_index, det_polygon in enumerate(det_polygons):
                if det_index in det_ignored or det_index in matched_det:
                    continue
                intersection = gt_polygon.intersection(det_polygon).area
                union = gt_polygon.union(det_polygon).area
                iou = 0.0 if union <= 0 else intersection / union
                if iou > self.iou_constraint:
                    matched_gt.add(gt_index)
                    matched_det.add(det_index)
                    matched += 1
                    break

        return {
            "gt_care": len(gt_polygons) - len(ignored_gt_indices),
            "det_care": len(det_polygons) - len(det_ignored),
            "matched": matched,
        }

    @staticmethod
    def combine_results(results):
        gt_care = sum(item["gt_care"] for item in results)
        det_care = sum(item["det_care"] for item in results)
        matched = sum(item["matched"] for item in results)
        recall = 0.0 if gt_care == 0 else matched / gt_care
        precision = 0.0 if det_care == 0 else matched / det_care
        hmean = (
            0.0
            if precision + recall == 0
            else 2.0 * precision * recall / (precision + recall)
        )
        return {"precision": precision, "recall": recall, "hmean": hmean}


class DetectionMetric:
    def __init__(self, post_process, main_indicator="hmean", evaluator=None):
        self.post_process = post_process
        self.main_indicator = str(main_indicator)
        if self.main_indicator not in {"precision", "recall", "hmean"}:
            raise ValueError(
                f"Unsupported detection main indicator: {self.main_indicator}"
            )
        self.evaluator = evaluator or DetectionIoUEvaluator()
        self.reset()

    @staticmethod
    def _shrink_map(outputs):
        if isinstance(outputs, dict):
            outputs = outputs["maps"]
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        if not torch.is_tensor(outputs) or outputs.ndim != 4:
            raise ValueError("Detection metric expects a 4D shrink-map tensor.")
        return outputs[:, :1]

    def update(self, outputs, targets):
        required = {"polygons", "ignore_tags", "shape"}
        missing = required - set(targets)
        if missing:
            raise ValueError(
                f"Detection metric targets are missing keys: {sorted(missing)}"
            )
        shrink_map = self._shrink_map(outputs).detach().float().cpu()
        predictions = self.post_process({"maps": shrink_map}, targets["shape"])
        if len(predictions) != len(targets["polygons"]):
            raise ValueError("Detection postprocess batch size does not match targets.")
        for prediction, polygons, ignore_tags in zip(
            predictions,
            targets["polygons"],
            targets["ignore_tags"],
        ):
            ground_truth = [
                {"points": points, "ignore": bool(ignored)}
                for points, ignored in zip(polygons, ignore_tags)
            ]
            detections = [
                {"points": points} for points in prediction.get("points", [])
            ]
            self.results.append(
                self.evaluator.evaluate_image(ground_truth, detections)
            )

    def compute(self):
        if not self.results:
            raise ValueError("Detection metric has no samples.")
        return self.evaluator.combine_results(self.results)

    def reset(self):
        self.results = []
