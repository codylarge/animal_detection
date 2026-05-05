from flask import Flask
from flask_cors import CORS
from routes.events import events_bp
from routes.media import media_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(events_bp)
app.register_blueprint(media_bp)

if __name__ == "__main__":
    app.run(debug=True, port=5000)