from app import create_app
import time, sys, cherrypy, os
from paste.translogger import TransLogger
from pyspark import SparkContext, SparkConf

def init_spark_context():
    conf = SparkConf().setAppName("yelp data recommender")
    sc = SparkContext(conf=conf, pyFiles=['recommender.py', 'app.py'])
    return sc

def run_server(app):
    app_logged = TransLogger(app)

    cherrypy.tree.graft(app_logged, '/')

    cherrypy.config.update({
        'engine.autoreload.on': True,
        'log.screen': True,
        'server.socket_port': 5000,
        'server.socket_host': '0.0.0.0'
    })

    cherrypy.engine.start()
    cherrypy.engine.block()

if __name__ == "__main__":
    sc = init_spark_context()
    app = create_app(sc)
    run_server(app)







