"""
SIH26166 — Adaptive Hybrid Lunar Registration — Web Frontend
-------------------------------------------------------------
Flask server that accepts two lunar images from the browser,
runs the *unchanged* registration algorithm, and returns
structured JSON results + the inlier-visualization image.
"""

import os
import uuid
import math
import json
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template

# Import the algorithm — completely untouched.
from .registration.pipeline import adaptive_register_with_rotation
from .registration.visualization import save_inlier_visualization

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(BASE_DIR.parent / "frontend" / "templates"),
    static_folder=str(BASE_DIR.parent / "frontend" / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(str(OUTPUT_DIR), filename)


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


@app.route("/run", methods=["POST"])
def run_registration():
    """Accept two images, run the algorithm, return JSON results."""
    if "image_a" not in request.files or "image_b" not in request.files:
        return jsonify({"error": "Both image_a and image_b are required."}), 400

    file_a = request.files["image_a"]
    file_b = request.files["image_b"]

    if file_a.filename == "" or file_b.filename == "":
        return jsonify({"error": "Please select both images."}), 400

    # Save uploads with unique names to avoid collisions.
    run_id = uuid.uuid4().hex[:10]
    ext_a = Path(file_a.filename).suffix or ".png"
    ext_b = Path(file_b.filename).suffix or ".png"
    path_a = UPLOAD_DIR / f"{run_id}_A{ext_a}"
    path_b = UPLOAD_DIR / f"{run_id}_B{ext_b}"
    file_a.save(str(path_a))
    file_b.save(str(path_b))

    # Read with OpenCV (exactly as the CLI main() does).
    img1 = cv2.imread(str(path_a), cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(str(path_b), cv2.IMREAD_UNCHANGED)

    if img1 is None:
        return jsonify({"error": f"Could not decode Image A ({file_a.filename})."}), 400
    if img2 is None:
        return jsonify({"error": f"Could not decode Image B ({file_b.filename})."}), 400

    # ---------- Run the algorithm (no changes) ----------
    result = adaptive_register_with_rotation(img1, img2)

    # Save inlier visualization.
    inlier_src = result.get("inlier_src", np.empty((0, 2), np.float32))
    inlier_dst = result.get("inlier_dst", np.empty((0, 2), np.float32))
    vis_name = f"inliers_{run_id}.png"
    vis_path = OUTPUT_DIR / vis_name
    save_inlier_visualization(
        img1,
        img2,
        inlier_src,
        inlier_dst,
        vis_path,
        f"{file_a.filename}  vs  {file_b.filename}",
        bool(result.get("accepted", False)),
    )

    # ---------- Build JSON-safe response ----------
    def safe(v):
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            return str(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v

    # Keys to show in the main results table (in order).
    TABLE_KEYS = [
        ("accepted",              "Registration Accepted"),
        ("stage",                 "Pipeline Stage"),
        ("registration_mode",    "Registration Mode"),
        ("rotation_used",        "Rotation Recovery Used"),
        ("rotation_supported",   "Rotation Supported"),
        ("rotation_angle_deg",   "Rotation Angle (°)"),
        ("rotation_confidence",  "Rotation Confidence"),
        ("seed_inliers",         "Seed Inliers"),
        ("seed_coverage",        "Seed Coverage"),
        ("seed_area_ratio",      "Seed Area Ratio"),
        ("seed_anisotropy",      "Seed Anisotropy"),
        ("seed_sane",            "Seed Sane"),
        ("recovery_used",        "AKAZE Recovery Used"),
        ("recovery_matches",     "Recovery Matches"),
        ("lk_points",            "LK Tracked Points"),
        ("final_inliers",        "Final Inliers"),
        ("final_rms",            "Final RMS (px)"),
        ("final_coverage",       "Final Coverage"),
        ("final_cells",          "Final Grid Cells"),
        ("final_area_ratio",     "Final Area Ratio"),
        ("final_anisotropy",     "Final Anisotropy"),
        ("final_geometry_ok",    "Final Geometry OK"),
        ("seed_runtime_s",       "Seed Runtime (s)"),
        ("lk_runtime_s",         "LK Runtime (s)"),
        ("rotation_runtime_s",   "Rotation Runtime (s)"),
        ("runtime_s",            "Total Runtime (s)"),
    ]

    table_rows = []
    for key, label in TABLE_KEYS:
        if key in result:
            val = result[key]
            # Format floats nicely.
            if isinstance(val, float) and not (math.isinf(val) or math.isnan(val)):
                if "runtime" in key or "rms" in key:
                    formatted = f"{val:.4f}"
                elif "ratio" in key or "coverage" in key or "confidence" in key:
                    formatted = f"{val:.4f}"
                elif "angle" in key:
                    formatted = f"{val:.2f}"
                else:
                    formatted = f"{val:.4f}"
            else:
                formatted = str(safe(val))
            table_rows.append({
                "key": key,
                "label": label,
                "value": formatted,
                "raw": safe(val),
            })

    response = {
        "table": table_rows,
        "accepted": bool(result.get("accepted", False)),
        "inlier_image": f"/outputs/{vis_name}",
        "image_a": f"/uploads/{path_a.name}",
        "image_b": f"/uploads/{path_b.name}",
        "image_a_name": file_a.filename,
        "image_b_name": file_b.filename,
        "image_a_shape": list(img1.shape),
        "image_b_shape": list(img2.shape),
    }

    return jsonify(response)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SIH26166 — Lunar Registration Frontend")
    print("  Open  http://127.0.0.1:5000  in your browser")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
