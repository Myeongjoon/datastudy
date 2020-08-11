from flask import Blueprint
import os
from flask import Flask
import hashlib
from recommender import Recommender

main = Blueprint('main', __name__)
base_dir = "./../data/"

def __get_model_path():
    model_file_name = "business_recomm_model"
    model_full_path = os.path.join(base_dir, "mf_based_models", model_file_name)
    return model_full_path

@main.route("/")
def index():
    return str(dir(recommender.spark_context))

@main.route("/user_id/<string:user_id>/top/<int:topk>/", methods=["GET"])
def get_recommended_businesses(user_id, topk):
    user_id = int(hashlib.sha1(user_id.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
    global current_city_name
    global my_spark_context

    load_models_and_businesses(my_spark_context)
        
    recommended_biz_ids = recommender.recommend_business_for_user(model, user_id, topk)

    return str(recommended_biz_ids)

def load_models_and_businesses(spark_context):
    global recommender
    global richer_biz_info
    global model

    model_path = __get_model_path()
    recommender = Recommender(spark_context, model_path)
    model = recommender.load_mf_model()

def create_app(spark_context):
    global my_spark_context

    my_spark_context = spark_context
    load_models_and_businesses(spark_context)

    app = Flask(__name__)
    app.register_blueprint(main)
    return app    
