from flask import Flask, request, render_template
from util.ner import get_entities

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    entities = []
    input_text = ""
    if request.method == "POST":
        input_text = request.form["text"]
        if input_text.strip():
            raw_ents = get_entities(input_text)
            for ent in raw_ents:
                if 'entity_group' in ent and 'word' in ent:
                    entities.append({
                        "word": ent["word"],
                        "entity_group": ent["entity_group"]
                    })
    return render_template("index.html", entities=entities, input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)