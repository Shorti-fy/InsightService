class SentimentClassifier:
    def __init__(self):
        import NewsSentiment
        from NewsSentiment import TargetSentimentClassifier
        self.tsc = TargetSentimentClassifier()
        
    def predict_sentiment(self, text):
        sentiment = self.tsc.infer_from_text("", text, "")
        ans=(sentiment[0]['class_label'])
        return ans