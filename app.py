import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient

load_dotenv()

app = Flask(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv(
    "MODEL_ID",
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({
            "success": False,
            "message": "Please provide some text."
        }), 400

    try:
        result = client.text_classification(
            text,
            model=MODEL_ID
        )

        # result is usually a list of labels/scores
        top = result[0] if isinstance(result, list) and result else result

        return jsonify({
            "success": True,
            "model": MODEL_ID,
            "input": text,
            "result": result,
            "top_label": getattr(top, "label", None) if not isinstance(top, dict) else top.get("label"),
            "top_score": getattr(top, "score", None) if not isinstance(top, dict) else top.get("score"),
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": "Inference failed.",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)