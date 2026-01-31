from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

alerts = []

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/alert", methods=["POST"])
def receive_alert():
    data = request.json
    alerts.append(data)
    print("ALERT RECEIVED:", data)
    return jsonify({"status": "received"})

@app.route("/alerts", methods=["GET"])
def get_alerts():
    return jsonify(alerts)

if __name__ == "__main__":
    app.run(debug=True)
