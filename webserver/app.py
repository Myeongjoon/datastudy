from flask import Blueprint
import os
from flask import Flask
import hashlib
from mf_based_recommender import MFBasedRecommender

main = Blueprint('main', __name__)
base_dir = "./../data/"

def __get_model_path():
    model_file_name = "business_recomm_model"
    model_full_path = os.path.join(base_dir, "mf_based_models", model_file_name)
    return model_full_path

def keywords_match_categories(keywords, categories):
    parsed_keywords = []
    for w in keywords:
        pw = w.strip().lower()
        if len(pw) != 0:
            parsed_keywords.append(pw)
    parsed_keywords = list(set(parsed_keywords))
    
    parsed_categories = [c.strip().lower() for c in categories]
    for kw in parsed_keywords:
        for c in parsed_categories:
            sp = c.split(" ")
            if len(sp) == 1:
                if kw in c:
                    return True
            else:
                if kw in sp:
                    return True
    return False

def get_rich_info_of_topk_businesses(richer_biz_info, business_ids, keywords, topk):
    lat_list = []
    lng_list = []
    review_count_list = []
    stars_list = []
    categories_list = []

    if len(keywords) == 0:
        keywords = ["Food"]

    for b in business_ids:
        r = richer_biz_info[richer_biz_info.integer_business_id == b]
        c = r.categories.values[0]
        c = eval(c)
        c = [str(s) for s in c]
        
        if keywords_match_categories(keywords, c):
            lat_list.append(r.latitude.values[0])
            lng_list.append(r.longitude.values[0])
            review_count_list.append(r.review_count.values[0])
            stars_list.append(r.stars.values[0])
            categories_list.append(c)

            if len(lat_list) == topk:
                break
        
    return (lat_list, lng_list, review_count_list, stars_list, categories_list)

@main.route("/")
def index():
    return str(dir(recommender.spark_context))

# http://0.0.0.0:5000/user_id/nIJD_7ZXHq-FX8byPMOkMQ/top/10

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
    recommender = MFBasedRecommender(spark_context, model_path)
    model = recommender.load_mf_model()

def create_app(spark_context):
    global my_spark_context

    my_spark_context = spark_context
    load_models_and_businesses(spark_context)

    app = Flask(__name__)
    app.register_blueprint(main)
    return app    
