FROM registry.cn-hangzhou.aliyuncs.com/gerry_ai/deploy_nlp:v20240624_01

# 将宿主机中的文件夹复制到镜像里面：要求宿主机里面的文件夹必须存在
COPY ner /mnt/ner

# 环境依赖安装
RUN /mnt/miniconda3/bin/pip install flask six -i http://mirrors.cloud.aliyuncs.com/pypi/simple/ --trusted-host mirrors.cloud.aliyuncs.com

# 这是环境变量
ENV MEDICAL_NER_MODEL_DIR /mnt/ner/ckpt

# 对外暴露的端口号
EXPOSE 9001

# 执行docker run的时候，默认执行命令
CMD /mnt/miniconda3/bin/python /mnt/ner/flask_app_starter.py
