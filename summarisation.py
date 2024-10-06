class Summarizer:
    def __init__(self):
      from transformers import AutoTokenizer, BartForConditionalGeneration
      self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
      self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    
    def summarise(self, text):
      inputs = self.tokenizer(text, max_length=1024, return_tensors="pt")
      summary_ids = self.model.generate(inputs["input_ids"], num_beams=2, min_length=100, max_length=250)
      summary= self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
      return summary