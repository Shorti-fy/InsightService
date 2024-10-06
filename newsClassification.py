class Classifier:
  def __init__(self):
    import google.generativeai as genai
    genai.configure(api_key='AIzaSyBFKAIxQmBZCmhaHWP1pGtLqAGxi88TboU')
    self.model= genai.GenerativeModel("gemini-1.5-flash")
  def classify(self, text):
    # pd.set_option("display.max_rows", 200)
    prompt= "You have the followin options: Business, Entertainment, Life & Style Society, Technology, Sport, Movies, Food, Education. \n GIVE ONLY ONE WORD ANSWER. CLAFFIFY THE FOLLOWING TEXT INTO ONLY 1 OF THE ABOVE OPTIONS: " + text
    response = self.model.generate_content(prompt)
    ans=response.text[:-2]
    return ans