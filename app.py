from flask import Flask, request, render_template_string
import pickle
import numpy as np

# 1. Create Flask app
app = Flask(__name__)

# 2. Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# 3. Simple HTML page (form)
html = """
<!doctype html>
<html>
  <head>
    <title>Marks Predictor</title>
  </head>
  <body>
    <h1>Marks Predictor</h1>
    <form method="post">
      <label>Hours studied:</label>
      <input type="number" step="any" name="hours" required>
      <button type="submit">Predict</button>
    </form>

    {% if prediction is not none %}
      <h2>Predicted marks: {{ prediction }}</h2>
    {% endif %}
  </body>
</html>
"""

# 4. Route for home page
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        hours = float(request.form["hours"])
        x = np.array([[hours]])
        pred = model.predict(x)[0]
        prediction = round(pred, 2)
    return render_template_string(html, prediction=prediction)

# 5. Run the app
if __name__ == "__main__":
    app.run(debug=True)
