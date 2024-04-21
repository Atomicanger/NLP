from flask import Flask, render_template, request
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration, pipeline

app = Flask(__name__)

# BART Models
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_summarization_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# T5 Models
t5_tokenizer = T5Tokenizer.from_pretrained("t5-large")
t5_summarization_model = T5ForConditionalGeneration.from_pretrained("t5-large")

def bart_summarize_text(text):
    input_ids = bart_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = bart_summarization_model.generate(input_ids, max_length=512,max_new_tokens=1024, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=False)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def t5_summarize_text(text):
    inputs = t5_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512,max_new_tokens=1024, truncation=True)
    summary_ids = t5_summarization_model.generate(inputs, max_length=512,max_new_tokens=1024, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=False)
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def bart_paraphrase_text(text):
    paraphrase_pipeline = pipeline("text2text-generation", model="facebook/bart-large-cnn")
    paraphrase = paraphrase_pipeline(text)[0]['generated_text']
    return paraphrase

def t5_paraphrase_text(text):
    paraphrase_pipeline = pipeline("text2text-generation", model="t5-large",legacy=False)
    paraphrase = paraphrase_pipeline(text)[0]['generated_text']
    return paraphrase

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    input_text = request.form['input_text']
    task = request.form['task']
    model = request.form['model']
    
    if task == 'summarize':
        if model == 'BART':
            result = bart_summarize_text(input_text)
        elif model == 'T5':
            result = t5_summarize_text(input_text)
        else:
            result = "Invalid model selection"
    elif task == 'paraphrase':
        if model == 'BART':
            result = bart_paraphrase_text(input_text)
        elif model == 'T5':
            result = t5_paraphrase_text(input_text)
        else:
            result = "Invalid model selection"
    else:
        result = "Invalid task selection"
    
    return render_template('index.html', input_text=input_text, result=result)

if __name__ == '__main__':
    app.run(debug=True)
