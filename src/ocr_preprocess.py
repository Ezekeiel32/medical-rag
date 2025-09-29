import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from PIL import Image
import cv2


@dataclass
class PreprocessConfig:
    target_max_megapixels: float = 20.0
    grayscale: bool = True
    orientation_4way: bool = True
    enable_deskew: bool = True
    enable_dewarp: bool = True
    enable_border_crop: bool = True
    enable_shadow_removal: bool = True
    enable_contrast_clahe: bool = True
    enable_denoise: bool = True
    enable_unsharp_mask: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    deskew_max_angle_deg: float = 15.0
    unsharp_amount: float = 1.2
    unsharp_radius_px: int = 1
    denoise_h: float = 7.0
    border_margin_ratio: float = 0.01


def _pil_to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _cv_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr)
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _maybe_downscale_max_mp(cvimg: np.ndarray, max_mp: float) -> Tuple[np.ndarray, float]:
    h, w = cvimg.shape[:2]
    mp = (w * h) / 1_000_000.0
    if mp <= max_mp:
        return cvimg, 1.0
    scale = (max_mp / mp) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(cvimg, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def _to_gray(cvimg: np.ndarray) -> np.ndarray:
    return cvimg if cvimg.ndim == 2 else cv2.cvtColor(cvimg, cv2.COLOR_BGR2GRAY)


def _binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )


def _horizontal_variance_score(bin_img: np.ndarray) -> float:
    row_sums = bin_img.sum(axis=1).astype(np.float64)
    var_score = float(np.var(row_sums)) / (bin_img.shape[1] * 255.0 + 1e-6)
    gx = cv2.Sobel(bin_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(bin_img, cv2.CV_32F, 0, 1, ksize=3)
    mag_x = float(np.mean(np.abs(gx)))
    mag_y = float(np.mean(np.abs(gy))) + 1e-6
    horiz_ratio = mag_x / (mag_x + mag_y)
    return var_score * 0.7 + horiz_ratio * 0.3


def _rotate_4way(cvimg: np.ndarray) -> Tuple[np.ndarray, int]:
    candidates = [
        (cvimg, 0),
        (cv2.rotate(cvimg, cv2.ROTATE_90_CLOCKWISE), 90),
        (cv2.rotate(cvimg, cv2.ROTATE_180), 180),
        (cv2.rotate(cvimg, cv2.ROTATE_90_COUNTERCLOCKWISE), 270),
    ]
    best_score = -1e9
    best_img = cvimg
    best_angle = 0
    for arr, ang in candidates:
        gray = _to_gray(arr)
        try:
            bin_img = _binarize(gray)
        except Exception:
            _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        score = _horizontal_variance_score(bin_img)
        if score > best_score:
            best_score = score
            best_img = arr
            best_angle = ang
    return best_img, best_angle


def _estimate_skew_angle(gray: np.ndarray, max_angle_deg: float) -> float:
    # Edge detect then Hough to find dominant near-horizontal lines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, threshold=150)
    if lines is None:
        return 0.0
    angles: List[float] = []
    for rho_theta in lines[:200]:
        rho, theta = rho_theta[0]
        # Convert to angle in degrees of line normal; text lines ~ horizontal => theta near 0 or pi
        angle = (theta - np.pi/2) * 180.0 / np.pi  # line angle relative to horizontal
        # Normalize to [-90, 90]
        while angle > 90:
            angle -= 180
        while angle < -90:
            angle += 180
        if abs(angle) <= max_angle_deg + 2:
            angles.append(angle)
    if not angles:
        return 0.0
    # Robust average
    angles_np = np.array(angles, dtype=np.float32)
    med = float(np.median(angles_np))
    return max(-max_angle_deg, min(max_angle_deg, med))


def _deskew(cvimg: np.ndarray, max_angle_deg: float) -> Tuple[np.ndarray, float]:
    gray = _to_gray(cvimg)
    ang = _estimate_skew_angle(gray, max_angle_deg)
    if abs(ang) < 0.5:
        return cvimg, 0.0
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
    rotated = cv2.warpAffine(cvimg, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated, ang


def _shadow_removal(gray: np.ndarray) -> np.ndarray:
    # Estimate background by dilating and median-blurring, then subtract
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(gray, bg)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return norm


def _apply_clahe(img: np.ndarray, clip: float, tile: int) -> np.ndarray:
    if img.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        return clahe.apply(img)
    # color: apply on L channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _denoise(img: np.ndarray, h: float) -> np.ndarray:
    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(img, h=h)
    return cv2.fastNlMeansDenoisingColored(img, hColor=h, h=h)


def _unsharp(img: np.ndarray, amount: float, radius_px: int) -> np.ndarray:
    if amount <= 0:
        return img
    if radius_px <= 0:
        return img
    blur = cv2.GaussianBlur(img, (radius_px*2+1, radius_px*2+1), 0)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def _largest_contour_box(bin_img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return x, y, w, h


def _border_crop(cvimg: np.ndarray, margin_ratio: float) -> np.ndarray:
    gray = _to_gray(cvimg)
    try:
        bin_img = _binarize(gray)
    except Exception:
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    box = _largest_contour_box(bin_img)
    if not box:
        return cvimg
    x, y, w, h = box
    H, W = gray.shape[:2]
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(W, x + w + mx)
    y2 = min(H, y + h + my)
    return cvimg[y1:y2, x1:x2]


def _order_points(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_warp(cvimg: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(cvimg, M, (maxW, maxH))


def _dewarp(cvimg: np.ndarray) -> np.ndarray:
    gray = _to_gray(cvimg)
    # Use edged image to detect quadrilateral page
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cvimg
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            try:
                warped = _four_point_warp(cvimg, pts)
                return warped
            except Exception:
                continue
    return cvimg


def preprocess_image(img: Image.Image, cfg: Optional[PreprocessConfig] = None) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Perform adaptive preprocessing on a PIL image for OCR.
    Returns (processed_pil, meta_dict).
    """
    if cfg is None:
        cfg = PreprocessConfig()
    meta: Dict[str, Any] = {"config": asdict(cfg)}

    cvimg = _pil_to_cv(img)
    meta["orig_shape"] = (int(cvimg.shape[0]), int(cvimg.shape[1]))

    # Downscale if needed
    try:
        cvimg, scale = _maybe_downscale_max_mp(cvimg, cfg.target_max_megapixels)
        meta["downscale_factor"] = float(scale)
    except Exception:
        meta["downscale_factor"] = 1.0

    # Orientation (4-way)
    if cfg.orientation_4way:
        try:
            cvimg, oang = _rotate_4way(cvimg)
            meta["orientation"] = int(oang)
        except Exception:
            meta["orientation"] = 0

    # Deskew
    if cfg.enable_deskew:
        try:
            cvimg, skew_deg = _deskew(cvimg, cfg.deskew_max_angle_deg)
            meta["skew_deg"] = float(skew_deg)
        except Exception:
            meta["skew_deg"] = 0.0

    # Dewarp (perspective)
    if cfg.enable_dewarp:
        try:
            cvimg = _dewarp(cvimg)
            meta["dewarped"] = True
        except Exception:
            meta["dewarped"] = False

    # Photometric: grayscale, shadow removal, CLAHE, denoise, unsharp
    if cfg.grayscale:
        try:
            gray = _to_gray(cvimg)
        except Exception:
            gray = cvimg if cvimg.ndim == 2 else cv2.cvtColor(cvimg, cv2.COLOR_BGR2GRAY)
    else:
        gray = None

    if cfg.enable_shadow_removal:
        try:
            if gray is None:
                g = _to_gray(cvimg)
            else:
                g = gray
            sr = _shadow_removal(g)
            gray = sr
            meta["shadow_removed"] = True
        except Exception:
            meta["shadow_removed"] = False

    if cfg.enable_contrast_clahe:
        try:
            if gray is not None:
                gray = _apply_clahe(gray, cfg.clahe_clip_limit, cfg.clahe_tile_grid_size)
            else:
                cvimg = _apply_clahe(cvimg, cfg.clahe_clip_limit, cfg.clahe_tile_grid_size)
            meta["clahe"] = True
        except Exception:
            meta["clahe"] = False

    if cfg.enable_denoise:
        try:
            if gray is not None:
                gray = _denoise(gray, cfg.denoise_h)
            else:
                cvimg = _denoise(cvimg, cfg.denoise_h)
            meta["denoise"] = True
        except Exception:
            meta["denoise"] = False

    if cfg.enable_unsharp_mask:
        try:
            if gray is not None:
                # convert to 3ch then back
                g3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                g3 = _unsharp(g3, cfg.unsharp_amount, cfg.unsharp_radius_px)
                gray = cv2.cvtColor(g3, cv2.COLOR_BGR2GRAY)
            else:
                cvimg = _unsharp(cvimg, cfg.unsharp_amount, cfg.unsharp_radius_px)
            meta["unsharp"] = True
        except Exception:
            meta["unsharp"] = False

    # Border crop
    if cfg.enable_border_crop:
        try:
            base = gray if gray is not None else cvimg
            # Use binarized image to detect page region
            if base.ndim != 2:
                base_gray = _to_gray(base)
            else:
                base_gray = base
            _ = _binarize(base_gray)
            # Crop original cvimg to retain color/grayscale choice
            cvimg = _border_crop(cvimg if gray is None else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cfg.border_margin_ratio)
            # If grayscale asked, keep gray aligned by recomputing from cropped
            if cfg.grayscale:
                gray = _to_gray(cvimg)
            meta["border_cropped"] = True
        except Exception:
            meta["border_cropped"] = False

    # Final compose
    if cfg.grayscale and gray is not None:
        out = gray
    else:
        out = cvimg

    pil = _cv_to_pil(out).convert("RGB")
    meta["final_shape"] = (int(out.shape[0]), int(out.shape[1]))
    return pil, meta


__all__ = ["PreprocessConfig", "preprocess_image"]