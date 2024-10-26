# -*- coding: utf-8 -*-
import json

import requests


def invoke_local():
    from medical_ner.deploy.predictor import Predictor

    # model_dir = "./ckpt/"
    model_dir = "./travel_query/"
    predictor = Predictor(
        model_dir=model_dir
    )
    result = predictor.predict(
        # "还有从湘潭到怀化的船票还有吗?"
        # "今年暑假，我准备去西安、新疆这个方向去玩，但是不知能能不能从上海坐火车过去?"
        "今年暑假，我准备去新疆玩，但是不知道能够从西安坐火车过去，我想先从上海坐飞机到西安，再慢慢的旅游过去"
        # "，患者2008年9月3日因“腹胀，发现腹部包块”在我院腹科行手术探查，术中见盆腹腔肿物，与肠管及子宫关系密切，遂行“全子宫左附件切除+盆腔肿物切除+右半结肠切除+DIXON术”，术后病理示颗粒细胞瘤，诊断为颗粒细胞瘤IIIC期，术后自2008年11月起行BEP方案化疗共4程，末次化疗时间为2009年3月26日。之后患者定期复查，2015-6-1，复查CT示：髂嵴水平上腹部L5腰椎前见软组织肿块，大小约30MM×45MM，密度欠均匀，边界尚清楚，轻度强化。查肿瘤标志物均正常。于2015-7-6行剖腹探查+膀胱旁肿物切除+骶前肿物切除+肠表面肿物切除术，术程顺利，，术后病理示：膀胱旁肿物及骶前肿物符合颗粒细胞瘤。于2015-7-13、8-14给予泰素240MG+伯尔定600MG化疗2程，过程顺利。出院至今，无发热，无腹痛、腹胀，有脱发，现返院复诊，拟行再次化疗收入院。起病以来，精神、胃纳、睡眠可，大小便正常，体重无明显改变。"
    )
    print(result)


def invoke_local_with_shell():
    from medical_ner.deploy.predictor import Predictor

    model_dir = "./ckpt/"
    # 初始化方法
    predictor = Predictor(model_dir=model_dir)

    while True:
        input_str = input("请输入患者病症描述:")
        if "q" == input_str.lower():
            break
        # 基于给定数据进行预测的方法
        result = predictor.predict(input_str)
        print(result, "\n\n")


def invoke_flask_web_api():
    response = requests.post(
        # url='http://118.31.246.133:9001/medical/ner',
        # url='http://121.196.193.3:9003/medical/ner',
        url='http://118.31.34.161:49153/medical/ner',
        data={
            'text': '，患者2008年9月3日因“腹胀，发现腹部包块”在我院腹科行手术探查，术中见盆腹腔肿物，与肠管及子宫关系密切，遂行“全子宫左附件切除+盆腔肿物切除+右半结肠切除+DIXON术”，术后病理示颗粒细胞瘤，诊断为颗粒细胞瘤IIIC期，术后自2008年11月起行BEP方案化疗共4程，末次化疗时间为2009年3月26日。之后患者定期复查，2015-6-1，复查CT示：髂嵴水平上腹部L5腰椎前见软组织肿块，大小约30MM×45MM，密度欠均匀，边界尚清楚，轻度强化。查肿瘤标志物均正常。于2015-7-6行剖腹探查+膀胱旁肿物切除+骶前肿物切除+肠表面肿物切除术，术程顺利，，术后病理示：膀胱旁肿物及骶前肿物符合颗粒细胞瘤。于2015-7-13、8-14给予泰素240MG+伯尔定600MG化疗2程，过程顺利。出院至今，无发热，无腹痛、腹胀，有脱发，现返院复诊，拟行再次化疗收入院。起病以来，精神、胃纳、睡眠可，大小便正常，体重无明显改变。'
        }
    )
    if response.status_code == 200:
        result = response.json()  # 将返回结果转换为json对象/dict对象
        print(result)
    else:
        print(f"访问异常 http code为: {response.status_code}")


def invoke_pai_eas():
    from eas_prediction import PredictClient
    from eas_prediction import StringRequest, StringResponse

    client = PredictClient('http://1757826125271350.cn-shenzhen.pai-eas.aliyuncs.com', 'medical_ner')
    client.set_token('MzdjNzhmMGNkMGQ4NzRkYWYzYWQxMTIwYjE4ZTBiYjljYTY4Y2E4Yg==')
    client.init()

    request = StringRequest('我是小明')
    request = StringRequest(
        "，患者2008年9月3日因“腹胀，发现腹部包块”在我院腹科行手术探查，术中见盆腹腔肿物，与肠管及子宫关系密切，遂行“全子宫左附件切除+盆腔肿物切除+右半结肠切除+DIXON术”，术后病理示颗粒细胞瘤，诊断为颗粒细胞瘤IIIC期，术后自2008年11月起行BEP方案化疗共4程，末次化疗时间为2009年3月26日。之后患者定期复查，2015-6-1，复查CT示：髂嵴水平上腹部L5腰椎前见软组织肿块，大小约30MM×45MM，密度欠均匀，边界尚清楚，轻度强化。查肿瘤标志物均正常。于2015-7-6行剖腹探查+膀胱旁肿物切除+骶前肿物切除+肠表面肿物切除术，术程顺利，，术后病理示：膀胱旁肿物及骶前肿物符合颗粒细胞瘤。于2015-7-13、8-14给予泰素240MG+伯尔定600MG化疗2程，过程顺利。出院至今，无发热，无腹痛、腹胀，有脱发，现返院复诊，拟行再次化疗收入院。起病以来，精神、胃纳、睡眠可，大小便正常，体重无明显改变。"
    )
    resp: StringResponse = client.predict(request)
    result = json.loads(resp.response_data)
    print(result)
    print(type(result))


if __name__ == '__main__':
    # invoke_local_with_shell()
    invoke_local()
    # invoke_flask_web_api()
    # invoke_pai_eas()
