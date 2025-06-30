# train/globals.py
class GlobalTracker:
    def __init__(self, model_scores):
        self.model_scores = model_scores

    def update(self, model_name, aunl_val):
        print(model_name)
        if aunl_val < self.model_scores[model_name]:
            self.model_scores[model_name] = aunl_val
            return True
        return False

    def get_score(self, model_name):
        return self.model_scores.get(model_name, float('inf'))

