# -*- coding: utf-8 -*-
import torch
from transformers import T5Config

from medical_ner.models.bert_crf_ner import AlBertCrfNerModel, BertCrfNerModel, T5CrfNerModel, T5CrfNerModel2


def t0():
    # https://huggingface.co/Tongjilibo/bert4torch_config/tree/main
    model = AlBertCrfNerModel(
        config_path=None,
        checkpoint_path="clue/albert_chinese_tiny",  # 实际上可以直接给定transformers里面的模型
        num_tags=3  # O、B-X、I-X
    )
    print(model)
    token_ids = torch.tensor([
        [101, 2532, 7856, 895, 102, 0, 0, 0],
        [101, 3565, 5685, 896, 352, 25, 521, 102]
    ])
    emission_score, attention_mask = model(token_ids)
    print(emission_score)
    print(attention_mask)


def tt_albert():
    from transformers import BertTokenizer
    from transformers import AlbertModel

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = AlbertModel.from_pretrained("clue/albert_chinese_tiny")

    texts = [
        'he like playing篮球',
        '还有双鸭山到淮阴的汽车票吗13号的',
        '从这里怎么回家'
    ]
    x = tokenizer(texts, padding=True)
    result = model(
        input_ids=torch.tensor(x['input_ids'], dtype=torch.int64),  # [N,T]
        attention_mask=torch.tensor(x['attention_mask'], dtype=torch.float32),  # [N,T]
        output_attentions=False,  # attention结果是否返回（每一个attention layer层的输出） attention
        output_hidden_states=True,  # 每一层的输出是否均返回 hidden_states
        return_dict=True  # 返回对象的类型是否是字典对象
    )
    print(result)


def tt_t5():
    from transformers import T5Tokenizer, T5Model, T5EncoderModel

    model_dir = "google-t5/t5-small"
    model_dir = "lemon234071/t5-base-Chinese"

    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    texts = [
        'he like playing篮球',
        '还有双鸭山到淮阴的汽车票吗13号的',
        '从这里怎么回家'
    ]
    x = tokenizer(texts, padding=True, return_tensors="pt")
    input_ids = tokenizer(
        "Studies have been shown that owning a dog is good for you", return_tensors="pt"
    ).input_ids  # Batch size 1
    decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids
    print(input_ids.shape)
    print(decoder_input_ids.shape)

    # model = T5Model.from_pretrained(model_dir)
    # print(model)
    # print(decoder_input_ids)
    # decoder_input_ids = model._shift_right(decoder_input_ids)
    # print(decoder_input_ids.shape)
    # print(decoder_input_ids)
    # result = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    # print(result.last_hidden_state.shape)

    # t5_encoder = T5EncoderModel.from_pretrained(model_dir)
    # t5_encoder = T5EncoderModel(T5Config.from_pretrained(model_dir))
    # result = t5_encoder(
    #     input_ids=torch.tensor(x['input_ids'], dtype=torch.int64),  # [N,T]
    #     # attention_mask=torch.tensor(x['attention_mask'], dtype=torch.float32),  # [N,T]
    # )
    # print(t5_encoder)
    # print(result.last_hidden_state.shape)

    model = T5CrfNerModel2(
        # config_path=r"D:\cache\huggingface\hub\models--google-t5--t5-small\snapshots\df1b051c49625cf57a3d0d8d3863ed4d13564fe4\config.json",
        # checkpoint_path="google-t5/t5-small",  # 实际上可以直接给定transformers里面的模型
        config_path=None,
        checkpoint_path=model_dir,
        num_tags=3  # O、B-X、I-X
    )
    print(model)
    token_ids = torch.tensor([
        [101, 2532, 7856, 895, 102, 0, 0, 0],
        [101, 3565, 5685, 896, 352, 25, 521, 102]
    ])
    emission_score, attention_mask = model(token_ids)
    print(emission_score)
    print(attention_mask)


if __name__ == '__main__':
    tt_t5()
