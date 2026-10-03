from flask import Flask, render_template, request, jsonify
import subprocess, sys, os, json
from datetime import datetime
from predict import FakeNewsPredictor
from preprocessing import preprocess_text

app = Flask(__name__)
predictor = FakeNewsPredictor()

# In-memory prediction history (resets on server restart)
prediction_history = []
MAX_HISTORY = 20


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() if request.is_json else {}
    news_text = data.get("text", "") or request.form.get("text", "")
    if not news_text:
        return jsonify({"error": "No text provided."}), 400

    result = predictor.predict(news_text)
    result["original_text"] = news_text

    if "prediction" in result:
        prediction_history.append({
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "timestamp": datetime.now().strftime("%H:%M"),
            "preview": news_text[:60] + ("..." if len(news_text) > 60 else ""),
        })
        if len(prediction_history) > MAX_HISTORY:
            prediction_history.pop(0)

    return jsonify(result)


@app.route("/history", methods=["GET"])
def history():
    """Return last 20 predictions for the history chart."""
    return jsonify(prediction_history)


@app.route("/analysis", methods=["POST"])
def analysis():
    """Return top words that pushed the prediction toward FAKE or REAL."""
    data = request.get_json() if request.is_json else {}
    text = data.get("text", "").strip()
    if not text or not predictor.model or not predictor.vectorizer:
        return jsonify({"fake_words": [], "real_words": []})

    clean = preprocess_text(text)
    vec = predictor.vectorizer.transform([clean])

    try:
        feature_names = predictor.vectorizer.get_feature_names_out()
        # VotingClassifier estimators order: pac=0, lr=1, rf=2, nb=3
        lr = predictor.model.estimators_[1]
        coef = lr.coef_[0]          # shape: (n_features,)  positive=REAL, negative=FAKE

        non_zero = vec.nonzero()[1]
        contributions = []
        for idx in non_zero:
            impact = float(vec[0, idx]) * float(coef[idx])
            contributions.append({"word": feature_names[idx], "impact": round(impact, 5)})

        contributions.sort(key=lambda x: x["impact"])

        fake_words = [{"word": c["word"], "weight": round(abs(c["impact"]), 5)}
                      for c in contributions[:8] if c["impact"] < 0]
        real_words = [{"word": c["word"], "weight": round(c["impact"], 5)}
                      for c in contributions[-8:][::-1] if c["impact"] > 0]

        return jsonify({"fake_words": fake_words, "real_words": real_words})
    except Exception as e:
        return jsonify({"fake_words": [], "real_words": [], "error": str(e)})


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json() if request.is_json else {}
    text = data.get("text", "").strip()
    correct_label = data.get("correct_label", "").strip().upper()
    if not text or correct_label not in ("REAL", "FAKE"):
        return jsonify({"error": "Invalid feedback payload."}), 400
    return jsonify(predictor.add_feedback(text, correct_label))


@app.route("/stats", methods=["GET"])
def stats():
    feedback_path = "models/feedback_data.json"
    if not os.path.exists(feedback_path):
        return jsonify({"total": 0, "real": 0, "fake": 0})
    with open(feedback_path, "r") as f:
        data = json.load(f)
    real_count = sum(1 for i in data if i.get("correct_label", "").upper() == "REAL")
    fake_count = sum(1 for i in data if i.get("correct_label", "").upper() == "FAKE")
    return jsonify({"total": len(data), "real": real_count, "fake": fake_count})


@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        result = subprocess.run(
            [sys.executable, "train_model.py"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=300,
        )
        if result.returncode == 0:
            predictor.load_models()
            return jsonify({"message": "Model retrained successfully!", "log": result.stdout[-2000:]})
        return jsonify({"error": "Training failed.", "log": result.stderr[-2000:]}), 500
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Training timed out."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
