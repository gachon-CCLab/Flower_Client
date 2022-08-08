# https://github.com/adap/flower/tree/main/examples/advanced_tensorflow 참조

import os, logging, json

import tensorflow as tf
import tensorflow_addons as tfa

import flwr as fl
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
# keras에서 내장 함수 지원(to_categofical())
from keras.utils.np_utils import to_categorical

import wandb

from functools import partial
import requests
from fastapi import FastAPI, BackgroundTasks
import asyncio
import uvicorn
from pydantic.main import BaseModel

# Log 포맷 설정
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__) 

# Make TensorFlow logs less verbose
# TF warning log 필터링
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# CPU만 사용
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# client pod number 추출
pod_name = os.environ['MY_POD_ID'].split('-')
client_num = int(pod_name[3]) # client 번호

# 성능지표 초기화
loss = 0
accuracy = 0
precision = 0
recall = 0
auc = 0
f1_score = 0
auprc=0

next_gl_model= 0 # 글로벌 모델 버전

# FL client 상태 확인
app = FastAPI()

# FL Client 상태 class
class FLclient_status(BaseModel):
    FL_client: int = client_num
    FL_client_online: bool = True
    FL_client_start: bool = False
    FL_client_fail: bool = False
    FL_server_IP: str = None 

status = FLclient_status()

# Define Flower client
class PatientClient(fl.client.NumPyClient):
    global client_num
    
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def get_properties(self, config):
        status = fl.common.Status(code=fl.common.Code.OK, message="Success")
        properties = {"mse": 0.5} # super().get_properties({"mse": 0.5})
        return fl.common.PropertiesRes(status, properties)
    
    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        num_rounds: int = config["num_rounds"]

        # wandb에 파라미터값 upload
        wandb.config.update({"num_rounds": num_rounds, "epochs": epochs,"batch_size": batch_size, "client_num": client_num})

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.2,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }

        # 매 round 마다 성능지표 확인을 위한 log
        loss = history.history["loss"][0]
        accuracy = history.history["accuracy"][0]
        precision = history.history["precision"][0]
        recall = history.history["recall"][0]
        auc = history.history["auc"][0]
        auprc = history.history["auprc"][0]
        f1_score = history.history["f1_score"][0]

        # print(history.history)

        # local model의 validation set  성능지표 wandb에 upload
        wandb.log({"loss": loss, "accuracy": accuracy, "precision": precision, "recall": recall, "auc":auc, "auprc":auprc,"f1_score": f1_score})        

        # local model의 validation set 성능지표 wandb에 upload
        # wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy, "val_precision": val_precision, "val_recall": val_recall, "val_auc":val_auc, "val_auprc":val_auprc, "val_f1_score": val_f1_score})

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        global status, loss, accuracy, precision, recall, auc, auprc, f1_score

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy, precision, recall, auc, auprc, f1_score = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)

        return loss, num_examples_test, {"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc, "auprc": auprc, "f1_score":f1_score}
        # return loss, num_examples_test, {"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc, 'f1_score': f1_score, 'auprc': auprc}

# Client Local Model 생성
def build_model():

    # 모델 및 메트릭 정의
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='auprc', curve='PR'), # precision-recall curve
        tfa.metrics.F1Score(name='f1_score', num_classes=10, average='micro'),
    ]

    # model 생성
    model = Sequential()

    # Convolutional Block (Conv-Conv-Pool-Dropout)
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Classifying
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=METRICS)

    return model

@app.on_event("startup")
def startup():
    pass

@app.get('/online')
def get_info():
    return status

@app.get("/start/{Server_IP}")
async def flclientstart(background_tasks: BackgroundTasks, Server_IP: str):
    global status, model, next_gl_model

    # client_manager 주소
    client_res = requests.get('http://localhost:8003/info/')

    # 최신 global model 버전
    latest_gl_model_v = client_res.json()['GL_Model_V']
    
    # 다음 global model 버전
    next_gl_model = latest_gl_model_v + 1

    logging.info('bulid model')

    logging.info('FL start')
    status.FL_client_start = True
    status.FL_server_IP = Server_IP
    background_tasks.add_task(flower_client_start)

    return status


async def flower_client_start():
    logging.info('FL learning')
    global model, status, x_train, y_train

    # 환자별로 partition 분리 => 개별 클라이언트 적용
    (x_train, y_train), (x_test, y_test) = load_partition()

    model = build_model()

    # wandb login and init
    wandb.login(key='6266dbc809b57000d78fb8b163179a0a3d6eeb37')
    wandb.init(entity='ccl-fl', project='fl-client', name= 'client %s_V%s'%(client_num,next_gl_model), dir='/')
    
    try:
        loop = asyncio.get_event_loop()
        client = PatientClient(model, x_train, y_train, x_test, y_test)
        # logging.info(f'fl-server-ip: {status.FL_server_IP}')
        # await asyncio.sleep(23)
        request = partial(fl.client.start_numpy_client, server_address=status.FL_server_IP, client=client)
        await loop.run_in_executor(None, request)

        logging.info('fl learning finished')
        await model_save()
        logging.info('model_save')
        del client, request

    except Exception as e:
        logging.info('[E][PC0002] learning', e)
        status.FL_client_fail = True
        await notify_fail()        
        status.FL_client_fail = False
        raise e
    return status

async def model_save():
    
    global model, next_gl_model
    try:
        model.save('/model/model_V%s.h5'%next_gl_model)
        await notify_fin()
        model=None
    except Exception as e:
        logging.info('[E][PC0003] learning', e)
        status.FL_client_fail = True
        await notify_fail()
        status.FL_client_fail = False

    return status

# client manager에서 train finish 정보 확인
async def notify_fin():
    global status, loss, accuracy, precision, recall, auc, auprc, f1_score, next_gl_model
    
    # wandb 종료
    wandb.finish()
    
    status.FL_client_start = False

    # 최종 성능 결과
    result = {"loss": loss, "accuracy": accuracy, "precision": precision, 
    "recall": recall, "auc": auc, "auprc": auprc, "f1_score": f1_score, "next_gl_model": next_gl_model}
    json_result = json.dumps(result)
    print(json_result)
    print('{"client_num": ' + str(client_num) + ', "log": "' + str(json_result)+'"}')

    loop = asyncio.get_event_loop()
    future2 = loop.run_in_executor(None, requests.get, 'http://localhost:8003/trainFin')
    r = await future2
    print('try notify_fin')
    if r.status_code == 200:
        print('trainFin')
    else:
        print('notify_fin error: ', r.content)
    return status

# client manager에서 train fail 정보 확인
async def notify_fail():
    global status

    # wandb 종료
    wandb.finish()

    logging.info('notify_fail start')

    status.FL_client_start = False
    loop = asyncio.get_event_loop()
    future1 = loop.run_in_executor(None, requests.get, 'http://localhost:8003/trainFail')
    r = await future1
    if r.status_code == 200:
        logging.info('trainFin')
    else:
        logging.info('notify_fail error: ', r.content)
    
    return status

def load_partition():
    # Load the dataset partitions
    global client_num

    # Cifar 10 데이터셋 불러오기
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # client_num 값으로 데이터셋 나누기
    (X_train, y_train) = X_train[client_num*100:(client_num+1)*500], y_train[client_num*100:(client_num+1)*500]
    (X_test, y_test) = 	X_test[client_num*100:(client_num+1)*500], y_test[client_num*100:(client_num+1)*500]

    # class 설정
    num_classes = 10

    # one-hot encoding class 범위 지정
    # Client마다 보유 Label이 다르므로 => 전체 label 수를 맞춰야 함
    train_labels = to_categorical(y_train, num_classes)
    test_labels = to_categorical(y_test, num_classes)

    # 전처리
    train_features = X_train.astype('float32') / 255.0
    test_features = X_test.astype('float32') / 255.0

    return (train_features, train_labels), (test_features, test_labels)

if __name__ == "__main__":
    try:
        # client api 생성 => client manager와 통신하기 위함
        uvicorn.run("app:app", host='0.0.0.0', port=8002, reload=True)
        
    finally:
        # FL client out
        requests.get('http://localhost:8003/flclient_out')
        logging.info('%s client close'%client_num)
