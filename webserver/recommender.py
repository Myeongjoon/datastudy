from pyspark.mllib.recommendation import MatrixFactorizationModel

class Recommender:

    def __init__(self, spark_context, model_path):
        self.spark_context = spark_context
        self.model_path = model_path

    def load_mf_model(self):
        loaded_model = MatrixFactorizationModel.load(self.spark_context, self.model_path)
        return loaded_model
        
    def recommend_business_for_user(self, model, user_id, topk=1000):
        ratings = model.recommendProducts(user_id, topk)
        return [int(r.product) for r in ratings]


