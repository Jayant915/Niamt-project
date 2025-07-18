from flask import Flask, request, render_template
<<<<<<< HEAD
<<<<<<< HEAD
from ner1 import get_entities  # ✅ Import from ner1.py
=======
from util.ner import get_entities
>>>>>>> de26ead47d097a515b194157e38f2dedc6206d28
=======
from util.ner import get_entities
>>>>>>> de26ead47d097a515b194157e38f2dedc6206d28

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
<<<<<<< HEAD
<<<<<<< HEAD
    app.run(debug=True)
=======
    app.run(debug=True)
>>>>>>> de26ead47d097a515b194157e38f2dedc6206d28
=======
    app.run(debug=True)
>>>>>>> de26ead47d097a515b194157e38f2dedc6206d28
