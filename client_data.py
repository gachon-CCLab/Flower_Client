
import itertools
import logging


import tensorflow as tf
import numpy as np

from keras.utils.np_utils import to_categorical


# Log 포맷 설정
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__) 

# pytorch로 변경 시 수정 필요 (Data format)
# Load the dataset partitions
def data_load(all_client_num, FL_client_num, dataset, skewed, balanced):

    # 데이터셋 불러오기
    if dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # class 설정
    num_classes = 10

    (X_train, y_train), (X_test, y_test) = data_partition(X_train, y_train, X_test, y_test, skewed, balanced, FL_client_num, all_client_num)
        
    # one-hot encoding class 범위 지정
    # Client마다 보유 Label이 다르므로 => 전체 label 수를 맞춰야 함
    train_labels = to_categorical(y_train, num_classes)
    test_labels = to_categorical(y_test, num_classes)

    # 전처리
    # train_features = X_train.astype('float32') / 255.0
    # test_features = X_test.astype('float32') / 255.0

    # return (train_features, train_labels), (test_features, test_labels)
    return (X_train, train_labels), (X_test, test_labels)


def data_partition(X_train, y_train, X_test, y_test, skewed, balanced, FL_client_num, all_client_num):

    np.random.seed(FL_client_num)
    
    # only balanced/Imbalanced
    if skewed == False:
        # Dataset size range for each FL Client
        train_range_size = int(len(y_train)/all_client_num)
        test_range_size = int(len(y_test)/all_client_num)
        
        train_first_size = train_range_size * FL_client_num
        test_first_size = test_range_size * FL_client_num
        
        train_next_size = train_first_size+train_range_size
        test_next_size = test_first_size+test_range_size
        
        if balanced == True:
            (X_train, y_train) = X_train[train_first_size:train_next_size], y_train[train_first_size:train_next_size]
            (X_test, y_test) = X_test[test_first_size:test_next_size], y_test[test_first_size:test_next_size]
            
        else: 
            # Imbalanced dataset
            # Random FL Client dataset size => Imbalanced
            train_size = np.random.randint(train_first_size, train_next_size)
            test_size = np.random.randint(test_first_size, test_next_size)

            (X_train, y_train) = X_train[train_first_size:train_size], y_train[train_first_size:train_size]
            (X_test, y_test) = X_test[train_first_size:train_size], y_test[train_first_size:train_size]
    
    # balanced skewed/ imbalanced skewed
    else:
        dataset='cifar10'
        skewed_spec='skewed two'

        (X_train, y_train), (X_test, y_test) = skewed_partition(X_train, y_train, X_test, y_test, skewed_spec, balanced, FL_client_num, all_client_num, dataset)

    return (X_train, y_train), (X_test, y_test)
    


# Imbalanced/one class Skewed
def skewed_partition(X_train, y_train, X_test, y_test, skewed, balanced, FL_client_num, all_client_num, dataset):

    np.random.seed(FL_client_num)
    
    if skewed == 'skewed one':
        labels = 6
    elif skewed == 'skewed two':
        labels = [6,9]
    elif skewed =='skewed three':
        labels = [6,9,0]
        
    # select label index
    train_indexs = []
    test_indexs = []
    for label in labels:
        train_indexs.append(np.where(y_train == label)[0])
        test_indexs.append(np.where(y_test == label)[0])
    
    # label index list화 및 정렬 => label 분포있게 추출 가능
    train_index_list = np.sort(list(itertools.chain(*train_indexs)))
    test_index_list = np.sort(list(itertools.chain(*test_indexs)))
    
    (X_train, y_train) = X_train[train_index_list], y_train[train_index_list]
    (X_test, y_test) = X_test[test_index_list], y_test[test_index_list]
        
    # Dataset size range for each FL Client
    train_range_size = int(len(y_train)/all_client_num)
    test_range_size = int(len(y_test)/all_client_num)

    train_first_size = train_range_size * FL_client_num
    test_first_size = test_range_size * FL_client_num
    
    train_next_size = train_first_size+train_range_size
    test_next_size = test_first_size+test_range_size

    
    if balanced == True:
        (X_train, y_train) = X_train[train_first_size:train_next_size], y_train[train_first_size:train_next_size]
        (X_test, y_test) = X_test[test_first_size:test_next_size], y_test[test_first_size:test_next_size]
        
    else: 
        # Imbalanced/Skewed dataset   
        # Random FL Client dataset size => Imbalanced
        train_size = np.random.randint(train_first_size, train_next_size)
        test_size = np.random.randint(test_first_size, test_next_size)

        (X_train, y_train) = X_train[train_first_size:train_size], y_train[train_first_size:train_size]
        (X_test, y_test) = X_test[test_first_size:test_size], y_test[test_first_size:test_size]

    if dataset == 'cifar10':
        pass
    
    else: # MNIST, FashionMNIST의 모델은 전이학습 모델이므로 3차원으로 설정
        # 28X28 -> 32X32
        # Pad with 2 zeros on left and right hand sides-
        X_train = np.pad(X_train[:,], ((0,0),(2,2),(2,2)), 'constant')
        X_test = np.pad(X_test[:,], ((0,0),(2,2),(2,2)), 'constant')


        # 배열의 형상을 변경해서 차원 수를 3으로 설정
        # # 전이학습 모델 input값 설정시 차원을 3으로 설정해줘야 함
        X_train = tf.expand_dims(X_train, axis=3, name=None)
        X_test = tf.expand_dims(X_test, axis=3, name=None)
        X_train = tf.repeat(X_train, 3, axis=3)
        X_test = tf.repeat(X_test, 3, axis=3)
    
    return (X_train, y_train), (X_test, y_test)