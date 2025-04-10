from flask import Flask, request, render_template
import pickle
import numpy as np
import random

app = Flask(__name__)

# Load your model and scaler (make sure the paths are correct)
model = pickle.load(open("model/best_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        input_features = [
            float(request.form["male"]),
            float(request.form["age"]),
            float(request.form["currentSmoker"]),
            float(request.form["cigsPerDay"]),
            float(request.form["BPMeds"]),
            float(request.form["prevalentStroke"]),
            float(request.form["prevalentHyp"]),
            float(request.form["diabetes"]),
            float(request.form["totChol"]),
            float(request.form["sysBP"]),
            float(request.form["diaBP"]),
            float(request.form["BMI"]),
            float(request.form["heartRate"]),
            float(request.form["glucose"]),
        ]

        # Reshape and scale input
        input_array = np.array([input_features])
        input_scaled = scaler.transform(input_array)

        # Predict
        prediction = model.predict(input_scaled)[0]
        age = float(request.form["age"])

        # Fun messages based on prediction
        if prediction == 1:
            if age <= 40:
                message = random.choice(
                    [
                        "They sent Batman to visit you.\n He just came to say goodbye.",
                        "You're too young for this 💔. Time to fix your habits.",
                        "Not even an adult and your heart's crying already 😓",
                        "High risk at this age? That's wild. Take this seriously.",
                        "Dude... get help. You’re not even out of school yet 😬",
                    ]
                )
            else:
                message = random.choice(
                    [
                        "Might be time to swap that butter chicken for a salad. 💔",
                        "Your heart's throwing shade. Time to see a doc!",
                        "Oof. Cardio isn’t optional anymore.",
                        "Yikes. You’re on thin ice with your heart.",
                        "Bro you’re too young to mess around with your heart like this 😬",
                        "Your heart’s not vibing with your choices.",
                        "Yikes. You’ve unlocked heart issues early access 😐",
                        "Even your heart’s like 'bruh'.",
                        "Your heart’s not vibing with your choices.",
                        "Yikes. You’ve unlocked heart issues early access 😐",
                        "Even your heart’s like 'bruh'.",
                    ]
                )
        else:
            message = random.choice(
                [
                    "😌 All good.\n But I’m just code written by someone who binged Python tutorials at 3AM,\n so... proceed with caution.",
                    "All clear! Your heart's still vibing. ❤️",
                    "Heart’s cool. You’re not dying *yet*. 😂",
                    "Your heart’s chill. Unlike your code.",
                    "You’re safe—for now. 😎",
                    "Green light from the heart squad. Keep it that way 💚",
                    "Heartbeat’s smooth. Can’t say the same about your diet tho 👀",
                    "No issues—your heart just dropped a 'like' 👍",
                    "Look at you, beating the odds—literally. 💪",
                    "All good... but don’t push your luck, chief. 🫡",
                    "Heart's in shape. Now go fix your sleep schedule. 🛌",
                    "Nice! Your heart's working overtime while you binge Netflix.",
                    "Clean report. But remember—no one stays young forever. 😬",
                ]
            )

        return render_template("index.html", prediction_message=message)

    except Exception as e:
        return (
            f"Something went wrong.\nNeed a good Engineer which is not you! : {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)
