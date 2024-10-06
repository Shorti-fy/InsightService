class TagGenerator:
    def __init__(self):
      import ast
    #   import pandas as pd
      import spacy
      import google.generativeai as genai
      import os
      genai.configure(api_key='AIzaSyBFKAIxQmBZCmhaHWP1pGtLqAGxi88TboU')
      self.tagger = spacy.load("en_core_web_sm")
      self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate_tags(self, text):
      import ast
    #   pd.set_option("display.max_rows", 200)
      doc = self.tagger(text)
      tags=[]
      for ent in doc.ents:
          tags.append(ent.text)
          # print(ent.text)
      tags= str(tags)
      prompt= tags + "This is the array of tags returned to me that i can use to label news articles. Remove the ones you think are nnot suitable titles for tags and ONLY RETURN THE FINAL OUTPUT ARRAY AND NOTHING ELSE!!"
      response = self.model.generate_content(prompt)
      list_data = ast.literal_eval(response.text.strip())
      return list_data