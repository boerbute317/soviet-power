# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

if 'nt' == os.name:
    os.environ['MEDICAL_NER_MODEL_DIR'] = os.path.abspath('./ckpt')

if __name__ == '__main__':
    from medical_ner.deploy.flask_app import app

    app.run(
        host="0.0.0.0",  # 0.0.0.0表示监听当前机器的所有IP地址
        port=9001  # 监听当前机器的哪个端口
    )
