from pyspark.mllib.recommendation import MatrixFactorizationModel

class MFBasedRecommender:

    def load_mf_model(self, spark_context, model_path):
        loaded_model = MatrixFactorizationModel.load(spark_context, model_path)
        return loaded_model
        
    def recommend_business_for_user(self, model, user_id, topk=100):
        return model.recommendProducts(user_id, topk)
