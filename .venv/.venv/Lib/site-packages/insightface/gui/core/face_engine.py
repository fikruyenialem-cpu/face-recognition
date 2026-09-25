"""InsightFace runtime wrapper used by the GUI."""

from __future__ import annotations

import glob
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .constants import AUTO_DET_SIZES, DEFAULT_DET_SIZE, DEFAULT_MODEL_NAME, DEFAULT_THRESHOLD
from .logging import get_logger
from .models import CompareResult, FaceRecord
from .quality import score_face
from .recognition import compare_embeddings, cosine_similarity, normalize_embedding
from .utils import crop_bbox

LOGGER = get_logger("face_engine")


class ModelNotLoadedError(RuntimeError):
    pass


class FaceEngine:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        providers: Optional[Iterable[str]] = None,
        det_size: tuple[int, int] = DEFAULT_DET_SIZE,
        root: str | os.PathLike[str] | None = None,
        custom_model_dir: str | os.PathLike[str] | None = None,
    ):
        self.model_name = model_name
        self.root = Path(os.path.expanduser(str(root or "~/.insightface")))
        self.custom_model_dir = Path(os.path.expanduser(str(custom_model_dir))) if custom_model_dir else None
        self.requested_providers = list(providers or ["CPUExecutionProvider"])
        self.det_size = det_size
        self.det_sizes = self._resolve_det_sizes(det_size)
        self.auto_det_size = len(self.det_sizes) > 1
        self.models: Dict[str, Any] = {}
        self.det_model = None
        self.loaded = False
        self.last_error = ""
        self.embedding_dim: Optional[int] = None
        self.last_latency_ms: Dict[str, float] = {}
        self.active_providers: List[str] = []
        self.ctx_id = -1
        self._prepared_det_size = None
        self._lock = threading.RLock()

    def is_loaded(self) -> bool:
        return self.loaded and self.det_model is not None

    def resolve_model_dir(self) -> Path:
        if self.custom_model_dir and self.custom_model_dir.exists():
            return self.custom_model_dir
        model_path = Path(os.path.expanduser(self.model_name))
        if model_path.exists() and model_path.is_dir():
            return model_path
        return self.root / "models" / self.model_name

    def load(self) -> None:
        with self._lock:
            self.loaded = False
            self.last_error = ""
            self.models.clear()
            start = time.perf_counter()
            model_dir = self.resolve_model_dir()
            if not model_dir.exists():
                self.last_error = (
                    f"Model directory not found: {model_dir}. Configure a local model directory "
                    "or install the model pack first."
                )
                LOGGER.warning(self.last_error)
                return
            onnx_files = sorted(glob.glob(str(model_dir / "*.onnx")))
            if not onnx_files:
                self.last_error = f"No .onnx model files found in {model_dir}."
                LOGGER.warning(self.last_error)
                return
            try:
                from insightface.app.common import Face
            except Exception as exc:
                self.last_error = f"InsightFace runtime import failed: {exc}"
                LOGGER.exception(self.last_error)
                return

            del Face
            try:
                for onnx_file in onnx_files:
                    model = self._load_model_from_onnx(onnx_file)
                    if model is None:
                        continue
                    if model.taskname in self.models:
                        continue
                    self.models[model.taskname] = model
                if "detection" not in self.models:
                    self.last_error = f"No detection model found in {model_dir}."
                    return
                self.det_model = self.models["detection"]
                ctx_id = 0 if any("CUDA" in provider for provider in self.requested_providers) else -1
                self.ctx_id = ctx_id
                for taskname, model in self.models.items():
                    if taskname == "detection":
                        self._prepare_detection()
                    else:
                        model.prepare(ctx_id)
                self.loaded = True
                self.embedding_dim = self._infer_embedding_dim()
                self.active_providers = self.requested_providers
                self.last_latency_ms["load"] = (time.perf_counter() - start) * 1000.0
            except Exception as exc:
                self.last_error = f"Model load failed: {exc}"
                LOGGER.exception(self.last_error)
                self.loaded = False

    def _load_model_from_onnx(self, onnx_file: str):
        """Load a model pack member, forcing GUI detection through SCRFD."""
        try:
            from insightface.model_zoo.arcface_onnx import ArcFaceONNX
            from insightface.model_zoo.attribute import Attribute
            from insightface.model_zoo.inswapper import INSwapper
            from insightface.model_zoo.landmark import Landmark
            from insightface.model_zoo.model_zoo import PickableInferenceSession
            from insightface.model_zoo.scrfd import SCRFD
        except Exception as exc:
            raise RuntimeError(f"InsightFace model imports failed: {exc}") from exc

        session_kwargs = {
            "providers": self.requested_providers,
            "provider_options": None,
        }
        session_options = self._quiet_onnxruntime_session_options()
        if session_options is not None:
            session_kwargs["sess_options"] = session_options
        session = PickableInferenceSession(onnx_file, **session_kwargs)
        LOGGER.info("Applied providers for %s: %s", onnx_file, getattr(session, "_providers", []))
        inputs = session.get_inputs()
        if not inputs:
            return None
        input_shape = inputs[0].shape
        outputs = session.get_outputs()
        input_height = self._shape_dim(input_shape, 2)
        input_width = self._shape_dim(input_shape, 3)

        if self._is_scrfd_detection_outputs(outputs):
            LOGGER.info("GUI detection model routed to SCRFD: %s", onnx_file)
            return SCRFD(model_file=onnx_file, session=session)
        if input_height == 192 and input_width == 192:
            return Landmark(model_file=onnx_file, session=session)
        if input_height == 96 and input_width == 96:
            return Attribute(model_file=onnx_file, session=session)
        if len(inputs) == 2 and input_height == 128 and input_width == 128:
            return INSwapper(model_file=onnx_file, session=session)
        if (
            input_height is not None
            and input_width is not None
            and input_height == input_width
            and input_height >= 112
            and input_height % 16 == 0
        ):
            return ArcFaceONNX(model_file=onnx_file, session=session)
        return None

    @staticmethod
    def _shape_dim(shape, index: int) -> Optional[int]:
        try:
            value = shape[index]
        except Exception:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_scrfd_detection_outputs(outputs) -> bool:
        return len(outputs) in {6, 9, 10, 15}

    @staticmethod
    def _quiet_onnxruntime_session_options():
        try:
            import onnxruntime

            onnxruntime.set_default_logger_severity(3)
            session_options = onnxruntime.SessionOptions()
            session_options.log_severity_level = 3
            return session_options
        except Exception:
            return None

    def _infer_embedding_dim(self) -> Optional[int]:
        for model in self.models.values():
            if getattr(model, "taskname", "") == "recognition":
                output_shape = getattr(model, "output_shape", None)
                if output_shape:
                    try:
                        return int(output_shape[-1])
                    except Exception:
                        return None
        return None

    def warmup(self) -> Dict[str, Any]:
        if not self.is_loaded():
            raise ModelNotLoadedError("Model is not loaded. Please open Models.")
        warmup_width = max(size[0] for size in self.det_sizes)
        warmup_height = max(size[1] for size in self.det_sizes)
        image = np.zeros((warmup_height, warmup_width, 3), dtype=np.uint8)
        start = time.perf_counter()
        self.detect_faces(image)
        elapsed = (time.perf_counter() - start) * 1000.0
        self.last_latency_ms["warmup"] = elapsed
        return {"warmup_ms": elapsed}

    def detect_faces(self, image: np.ndarray, source_path: Optional[str] = None) -> List[FaceRecord]:
        with self._lock:
            if not self.is_loaded():
                raise ModelNotLoadedError("Model is not loaded. Please open Models.")
            if image is None:
                return []
            image = np.ascontiguousarray(image)
            try:
                from insightface.app.common import Face
            except Exception as exc:
                raise ModelNotLoadedError(f"InsightFace Face class import failed: {exc}") from exc

            start = time.perf_counter()
            bboxes, kpss = self._detect_raw(image)
            records: List[FaceRecord] = []
            if bboxes is None or bboxes.shape[0] == 0:
                self.last_latency_ms["detect"] = (time.perf_counter() - start) * 1000.0
                return records
            for idx in range(bboxes.shape[0]):
                bbox = bboxes[idx, 0:4]
                det_score = float(bboxes[idx, 4])
                kps = kpss[idx] if kpss is not None else None
                face = Face(bbox=bbox, kps=kps, det_score=det_score)
                for taskname, model in self.models.items():
                    if taskname == "detection":
                        continue
                    model.get(image, face)
                embedding = getattr(face, "embedding", None)
                normed_embedding = getattr(face, "normed_embedding", None)
                if normed_embedding is None and embedding is not None:
                    normed_embedding = normalize_embedding(embedding)
                crop = crop_bbox(image, bbox)
                quality_score, quality_flags = score_face(image, bbox, kps, det_score)
                records.append(
                    FaceRecord(
                        face_id=f"{source_path or 'image'}:{idx}",
                        bbox=[float(v) for v in bbox],
                        kps=kps.tolist() if kps is not None else None,
                        det_score=det_score,
                        embedding=np.asarray(embedding, dtype=np.float32).reshape(-1) if embedding is not None else None,
                        normed_embedding=(
                            np.asarray(normed_embedding, dtype=np.float32).reshape(-1)
                            if normed_embedding is not None
                            else None
                        ),
                        gender=getattr(face, "gender", None),
                        age=getattr(face, "age", None),
                        quality_score=quality_score,
                        quality_flags=quality_flags,
                        crop=crop,
                        source_path=source_path,
                    )
                )
            self.last_latency_ms["detect"] = (time.perf_counter() - start) * 1000.0
            return records

    def detect_best_face(self, image: np.ndarray, source_path: Optional[str] = None) -> Optional[FaceRecord]:
        faces = self.detect_faces(image, source_path=source_path)
        if not faces:
            return None
        return max(
            faces,
            key=lambda face: (
                (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
                face.det_score,
            ),
        )

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return cosine_similarity(emb1, emb2)

    def compare_images(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        threshold: float = DEFAULT_THRESHOLD,
        path1: Optional[str] = None,
        path2: Optional[str] = None,
    ) -> CompareResult:
        with self._lock:
            face_a = self.detect_best_face(image1, source_path=path1)
            face_b = self.detect_best_face(image2, source_path=path2)
            if face_a is None:
                raise ValueError("No face detected in Image A.")
            if face_b is None:
                raise ValueError("No face detected in Image B.")
            if face_a.normed_embedding is None or face_b.normed_embedding is None:
                raise ValueError("Recognition embedding is unavailable for one or both faces.")
            comparison = compare_embeddings(face_a.normed_embedding, face_b.normed_embedding, threshold)
            notes = []
            for label, face in (("Image A", face_a), ("Image B", face_b)):
                if face.quality_score is not None and face.quality_score < 0.45:
                    notes.append(f"{label} has low quality. Recognition may be unreliable.")
                if face.quality_flags:
                    notes.append(f"{label}: {', '.join(face.quality_flags)}")
            return CompareResult(
                similarity=float(comparison["similarity"]),
                threshold=threshold,
                decision=str(comparison["decision"]),
                face_a=face_a,
                face_b=face_b,
                notes=notes,
            )

    def get_runtime_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_dir": str(self.resolve_model_dir()),
            "providers": self.active_providers or self.requested_providers,
            "detector": type(self.det_model).__name__ if self.det_model is not None else None,
            "det_size": self._det_size_label(),
            "embedding_dim": self.embedding_dim,
            "loaded": self.is_loaded(),
            "last_error": self.last_error,
            "last_latency_ms": self.last_latency_ms,
        }

    def _resolve_det_sizes(self, det_size: tuple[int, int]) -> list[tuple[int, int]]:
        width, height = int(det_size[0]), int(det_size[1])
        if width <= 0 or height <= 0:
            return [(int(width), int(height)) for width, height in AUTO_DET_SIZES]
        return [(width, height)]

    def _det_size_label(self) -> str:
        if self.auto_det_size:
            return "Auto (" + " + ".join(f"{width}x{height}" for width, height in self.det_sizes) + ")"
        width, height = self.det_sizes[0]
        return f"{width}x{height}"

    def _prepare_detection(self) -> None:
        if self.det_model is None:
            return
        input_size = self._detector_input_size()
        if self._prepared_det_size == input_size:
            return
        LOGGER.debug("Preparing detection model with det_size=%s", input_size)
        self.det_model.prepare(self.ctx_id, input_size=input_size, det_thresh=0.5)
        self._prepared_det_size = input_size

    def _detect_raw(self, image: np.ndarray):
        self._prepare_detection()
        return self.det_model.detect(image, max_num=0, metric="default")

    def _detector_input_size(self):
        if self.auto_det_size:
            return list(self.det_sizes)
        return self.det_sizes[0]

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.4) -> list[int]:
        if boxes.size == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1 + 1.0) * np.maximum(0.0, y2 - y1 + 1.0)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[current], x1[order[1:]])
            yy1 = np.maximum(y1[current], y1[order[1:]])
            xx2 = np.minimum(x2[current], x2[order[1:]])
            yy2 = np.minimum(y2[current], y2[order[1:]])
            width = np.maximum(0.0, xx2 - xx1 + 1.0)
            height = np.maximum(0.0, yy2 - yy1 + 1.0)
            inter = width * height
            union = areas[current] + areas[order[1:]] - inter
            iou = np.where(union > 0, inter / union, 0.0)
            order = order[np.where(iou <= iou_threshold)[0] + 1]
        return keep


def available_execution_providers() -> List[str]:
    try:
        import onnxruntime

        providers = onnxruntime.get_available_providers()
        return [str(provider) for provider in providers]
    except Exception as exc:
        LOGGER.debug("Unable to inspect ONNX Runtime providers: %s", exc)
        return []


def is_cuda_provider_available() -> bool:
    return "CUDAExecutionProvider" in available_execution_providers()


def providers_from_choice(choice: str) -> List[str]:
    normalized = (choice or "CPU").strip().lower()
    if normalized == "cpu":
        return ["CPUExecutionProvider"]
    if normalized == "cuda":
        if is_cuda_provider_available():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        LOGGER.warning("CUDA provider was requested but CUDAExecutionProvider is not available; using CPU.")
        return ["CPUExecutionProvider"]
    available = available_execution_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
