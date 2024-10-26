# -*- coding: utf-8 -*-
"""
Flask是python的一种微服务框架，能够对外提供Http的服务接口
    安装方式: pip install flask
    参考:
        https://dormousehole.readthedocs.io/en/latest/
        https://github.com/pallets/flask/
        https://flask.palletsprojects.com/en/3.0.x/
"""
import logging
import os

from flask import request, jsonify, Flask

from .predictor import Predictor

app = Flask(__name__)
app.json.ensure_ascii = False
predictor = Predictor(model_dir=os.environ['MEDICAL_NER_MODEL_DIR'])


@app.route("/")
def index():
    return "欢迎使用Flask搭建web后端框架!"


@app.route("/medical/ner", methods=['GET', 'POST'])
def predict_ner():
    try:
        if request.method == 'GET':
            text = request.args.get('text')
        elif request.method == 'POST':
            text = request.form.get('text')
        else:
            raise ValueError(f"当前不支持该请求方式:{request.method}")
        if text is None or len(text) == 0:
            return jsonify({'code': 1, 'msg': 'request text param is null.'})
        result = predictor.predict(text)
        return jsonify({'code': 0, 'msg': 'successful.', 'data': result})
    except Exception as e:
        logging.error(f"NER后端接口执行异常:{e}", exc_info=e)
        return jsonify({'code': 2, 'msg': 'server error.'})
