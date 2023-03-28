import os
from flask import Flask, render_template, request, jsonify
import numpy as np
from transformers import TextClassificationPipeline
from transformers import BertForSequenceClassification

np.random.seed(1337)

app = Flask(__name__)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=3,
                                                      output_attentions=False,
                                                      output_hidden_states=False)
model.load_state_dict(torch.load("model/finetuned_BERT_epoch_3.model"))

def get_label(text):
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False,device="cuda:0")
    output=pipe(text)[0]['label']
    if output=='LABEL_0':
        label="Compte"
    elif output=='LABEL_1':
        label="Monétique"
    elif output=='LABEL_2':
        label="Multicanal"
    else:
        label="-1"
    return label

@app.route('/', methods=['GET', 'POST'])
def predict():
    """ Prediction function."""
    response = request.form['text']
    prediction_tag=get_label(response)
        # Render the request to the template
        return render_template('index.html', text=prediction_tag, submission=response)

    if request.method == 'GET':
        return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000,debug=False)