# train/globals.py
class GlobalTracker:
    def __init__(self, model_scores):
        self.model_scores = model_scores

    # Returns (aunl_improved, metric_improved)
    def update_aunl(self, model_name, aunl):
        if aunl < self.model_scores[model_name]['aunl']:
            self.model_scores[model_name]['aunl'] = aunl

    def update_metric(self, model_name, metric):
        if metric < self.model_scores[model_name]['metric']:
            self.model_scores[model_name]['metric'] = metric

    def get_score(self, model_name, which='aunl'):
        return self.model_scores.get(model_name, {}).get(which, float('inf'))

GLOBAL_PATIENCE = 2
MIN_EPOCHS = 5
TRACKING_METRIC = 'MAE'