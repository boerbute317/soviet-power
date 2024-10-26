# -*- coding: utf-8 -*-
import json
import logging
import os
import sys

import allspark

CODE_PATH = os.environ.get('MEDICAL_NER_CODE_DIR')
if CODE_PATH is not None:
    sys.path.append(CODE_PATH)


class MyProcessor(allspark.BaseProcessor):
    """ MyProcessor is a example
        you can send mesage like this to predict
        curl -v http://127.0.0.1:8080/api/predict/service_name -d '2 105'
    """

    def initialize(self):
        """
            load module, executed once at the start of the service
             do service intialization and load models in this function.
        """
        from medical_ner.deploy.predictor import Predictor

        model_dir = os.path.abspath(os.environ['MEDICAL_NER_MODEL_DIR'])
        print(f"模型文件夹为:{model_dir}")

        self.predictor = Predictor(model_dir=model_dir)

    def process(self, data):
        """ process the request data
        """
        try:
            text = str(data, encoding='utf-8')
            result = json.dumps(
                {
                    'code': 0,
                    'msg': 'successful',
                    'data': self.predictor.predict(text)
                },
                ensure_ascii=False
            )
            return bytes(result, encoding='utf8'), 200
        except Exception as e:
            logging.error(f"服务器异常:{e}", exc_info=e)
            result = json.dumps({'code': 1, 'msg': '服务器异常'}, ensure_ascii=False)
            return bytes(result, encoding='utf8'), 200


if __name__ == '__main__':
    # parameter worker_threads indicates concurrency of processing
    endpoint = os.environ['PAI_ENDPOINT']
    runner = MyProcessor(worker_threads=10, endpoint=endpoint)
    runner.run()
