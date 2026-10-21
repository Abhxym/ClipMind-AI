"""
OpenCV Tool — AI Video Director
================================
Utility functions for frame-level computer vision operations.
Used by: preprocess_agent, scene_agent, metadata_agent
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def compute_blur_score(frame_path: str) -> float:
    """
    Compute Laplacian variance (blur score) for a frame.
    Higher = sharper. Typical threshold for 'in-focus': > 100.
    """
    try:
        import cv2
        img = cv2.imread(frame_path)
        if img is None:
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except ImportError:
        return -1.0
    except Exception:
        return 0.0


def compute_brightness(frame_path: str) -> float:
    """Compute mean brightness (0-255) of a grayscale frame."""
    try:
        import cv2
        img = cv2.imread(frame_path)
        if img is None:
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(gray.mean())
    except ImportError:
        return -1.0
    except Exception:
        return 0.0


def compute_motion_score(frame1_path: str, frame2_path: str) -> float:
    """
    Compute optical-flow motion score between two consecutive frames.
    Returns mean magnitude of motion vectors (0 = static, higher = more motion).
    """
    try:
        import cv2
        import numpy as np

        img1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            return 0.0

        # Resize for speed
        img1 = cv2.resize(img1, (320, 240))
        img2 = cv2.resize(img2, (320, 240))

        flow = cv2.calcOpticalFlowFarneback(
            img1, img2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0,
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(magnitude.mean())
    except ImportError:
        return -1.0
    except Exception:
        return 0.0


def get_dominant_colors(frame_path: str, n_colors: int = 5) -> list:
    """
    Extract dominant color descriptors using K-means clustering.
    Returns list of (r, g, b) tuples.
    """
    try:
        import cv2
        import numpy as np

        img = cv2.imread(frame_path)
        if img is None:
            return []
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixels = img_rgb.reshape(-1, 3).astype(np.float32)

        k = min(n_colors, len(pixels))
        _, labels, centers = cv2.kmeans(
            pixels, k,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
            10, cv2.KMEANS_RANDOM_CENTERS
        )
        # Sort by frequency
        unique, counts = np.unique(labels, return_counts=True)
        sorted_centers = centers[unique[np.argsort(-counts)]]
        return [(int(c[0]), int(c[1]), int(c[2])) for c in sorted_centers]
    except ImportError:
        return []
    except Exception:
        return []


def is_frame_usable(
    frame_path: str,
    blur_threshold: float = 100.0,
    brightness_min: float = 40.0,
    brightness_max: float = 220.0,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Combined quality check for a frame.
    Returns (is_usable, metrics_dict).
    """
    blur = compute_blur_score(frame_path)
    brightness = compute_brightness(frame_path)

    if blur < 0:
        # cv2 not installed — assume usable
        return True, {"blur": blur, "brightness": brightness, "cv2_available": False}

    usable = (blur >= blur_threshold) and (brightness_min <= brightness <= brightness_max)
    return usable, {
        "blur": round(blur, 2),
        "brightness": round(brightness, 2),
        "cv2_available": True,
        "blur_ok": blur >= blur_threshold,
        "brightness_ok": brightness_min <= brightness <= brightness_max,
    }
