from flask import Blueprint, send_file, jsonify, abort
import os

media_bp = Blueprint("media", __name__)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "motion_snaps")

@media_bp.route("/api/media/video/<path:folder>", methods=["GET"])
def get_video(folder):
    video_path = os.path.join(BASE_DIR, folder, "event.mp4")
    if not os.path.exists(video_path):
        abort(404)
    return send_file(video_path, mimetype="video/mp4")

@media_bp.route("/api/media/images/<path:folder>", methods=["GET"])
def get_images(folder):
    folder_path = os.path.join(BASE_DIR, folder)
    if not os.path.exists(folder_path):
        abort(404)
    images = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
    return jsonify(images)