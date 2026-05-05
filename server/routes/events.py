from flask import Blueprint, jsonify
from database import get_db

events_bp = Blueprint("events", __name__)

@events_bp.route("/api/events", methods=["GET"])
def get_events():
    conn = get_db()
    events = conn.execute(
        "SELECT * FROM events ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return jsonify([dict(e) for e in events])

@events_bp.route("/api/events/<int:event_id>", methods=["GET"])
def get_event(event_id):
    conn = get_db()
    event = conn.execute(
        "SELECT * FROM events WHERE id = ?", (event_id,)
    ).fetchone()
    conn.close()
    if event is None:
        return jsonify({"error": "Event not found"}), 404
    return jsonify(dict(event))