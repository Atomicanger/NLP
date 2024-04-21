from transformers import BartTokenizer, BartForConditionalGeneration

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Save the tokenizer and model to a local directory
tokenizer.save_pretrained("bart_model")
model.save_pretrained("bart_model")
