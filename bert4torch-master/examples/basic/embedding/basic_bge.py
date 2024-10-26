# BAAI--bge-large-zh-v1.5
# BAAI--bge-large-en-v1.5
# BAAI--bge-base-zh-v1.5
# BAAI--bge-base-en-v1.5
# BAAI--bge-small-zh-v1.5
# BAAI--bge-small-en-v1.5
root_model_path = '/data/pretrain_ckpt/embedding/BAAI--bge-base-zh-v1.5'

sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]

print('=========================================sentence transformer====================================')
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(root_model_path)
embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)


print('=========================================bert4torch====================================')
from bert4torch.pipelines import Text2Vec
text2vec = Text2Vec(checkpoint_path=root_model_path, device='cuda')
embeddings_1 = text2vec.encode(sentences_1, normalize_embeddings=True)
embeddings_2 = text2vec.encode(sentences_2, normalize_embeddings=True)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
