import os
import urllib
import uuid
import logging

from PIL import Image
from flask import Flask, render_template, request
from keras.models import load_model
from keras.utils import load_img, img_to_array
from urllib.error import URLError
from werkzeug.utils import secure_filename

app = Flask("ClassificationWebApp")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = load_model(os.path.join(BASE_DIR, "modeling/model.hdf5"))
TARGET_IMG = os.path.join(BASE_DIR, "static/images")
ALLOWED_EXT: set = {"jpg", "jpeg", "png", "jfif"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1] in ALLOWED_EXT


def handle_url_upload(link: str) -> tuple[str, str]:
    try:
        resource = urllib.request.urlopen(link)
        unique_filename = str(uuid.uuid4())
        filename = unique_filename + ".jpg"
        img_path = os.path.join(TARGET_IMG, filename)
        output = open(img_path, "wb")
        output.write(resource.read())
        output.close()

        return img_path, ""

    except (URLError, ValueError) as e:
        logging.error(str(e))
        error = "Invalid URL or the image is not accessible"
        return "", error


def handle_file_upload(file) -> tuple[str, str]:
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(TARGET_IMG, filename))
        img_path = os.path.join(TARGET_IMG, filename)

        return img_path, ""

    else:
        error = "Please upload images of jpg, jpeg, and png extensions only"
        return "", error


def predict(filename: str, model) -> tuple[list[str], list[float]]:
    img = load_img(filename, target_size=(32, 32))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)

    img = img.astype("float32")
    img = img / 255.0
    result = model.predict(img)

    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    dict_result = {}
    for i in range(len(classes)):
        dict_result[result[0][i]] = classes[i]

    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[0]

    prob_result = [(prob * 100).round(2)]
    class_result = [dict_result[prob]]

    return class_result, prob_result


@app.route("/")
def home() -> str:
    return render_template("index.html", error="")


@app.route("/success", methods=["GET", "POST"])
def success() -> str:
    error = ""

    if request.method == "POST":
        if request.form:
            link = request.form.get("link")
            if link.strip() != "":
                img_path, error = handle_url_upload(link)
            else:
                error = "Please enter a valid URL"
        elif request.files:
            file = request.files["file"]
            img_path, error = handle_file_upload(file)
        else:
            error = "Invalid request"

        if error:
            return render_template("index.html", error=error)

        img = os.path.basename(img_path)
        class_result, prob_result = predict(img_path, MODEL)

        predictions = {
            "class1": class_result[0],
            "prob1": prob_result[0],
        }

        return render_template(
            "success.html", img=img, predictions=predictions, error=""
        )

    else:
        return render_template("index.html", error="")


def run_flask_app() -> None:
    app.run(debug=True)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(filename="app.log", level=logging.ERROR)

    # Start Flask app
    run_flask_app()
