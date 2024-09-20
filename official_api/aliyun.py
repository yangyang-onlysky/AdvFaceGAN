# -*- coding: utf-8 -*-
from alibabacloud_facebody20191230.models import CompareFaceAdvanceRequest, CreateFaceDbRequest, ListFaceDbsRequest, \
    AddFaceEntityRequest, GetFaceEntityRequest, ListFaceEntitiesRequest, AddFaceAdvanceRequest, \
    BatchAddFacesAdvanceRequest, BatchAddFacesAdvanceRequestFaces, DeleteFaceRequest, DeleteFaceEntityRequest, \
    DeleteFaceDbRequest, SearchFaceAdvanceRequest
from alibabacloud_tea_openapi.models import Config
from alibabacloud_facebody20191230.client import Client
from alibabacloud_tea_util.models import RuntimeOptions

import os
import time
import json
from tqdm import tqdm

config = Config(
    access_key_id=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_ID'),
    access_key_secret=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
    endpoint='facebody.cn-shanghai.aliyuncs.com',
    region_id='cn-shanghai'
)
# 初始化Client
client = Client(config)

qps=2

# 人脸比对
def face_compare(face1_path, face2_path):
    compare_face_request = CompareFaceAdvanceRequest()
    streamA = open(face1_path, 'rb')
    streamB = open(face2_path, 'rb')
    compare_face_request.image_urlaobject = streamA
    compare_face_request.image_urlbobject = streamB
    try:
        runtime_option = RuntimeOptions()
        response = client.compare_face_advance(compare_face_request, runtime_option)
        time.sleep(1.0 / qps)
        # 获取整体结果
        return response.body.to_map()['Data']['Confidence']
    except Exception as error:
        # 获取整体报错信息
        print(error)
    finally:
        # 关闭流
        streamA.close()
        streamB.close()
    return None

# 人脸搜索
def search_face(face_path, dbName, limit=5,  max_face_num=5, quality_score_threshold=50):
    stream = open(face_path, 'rb')
    search_face_request = SearchFaceAdvanceRequest(image_url_object=stream,
                                                   db_name=dbName,
                                                   limit=limit,
                                                   max_face_num=max_face_num,
                                                   quality_score_threshold=quality_score_threshold)
    try:
        runtime_option = RuntimeOptions()
        response = client.search_face_advance(search_face_request, runtime_option)
        time.sleep(1.0 / qps)
        # 获取整体结果
        return response.body.to_map()['Data']['MatchList']
    except Exception as error:
        # 获取整体报错信息
        print(error)
    finally:
        # 关闭流
        stream.close()


# 创建人脸数据库
def create_face_db(dbName):
    create_face_db_request = CreateFaceDbRequest(name=dbName)
    try:
        runtime_option = RuntimeOptions()
        response = client.create_face_db_with_options(create_face_db_request, runtime_option)
        time.sleep(1.0 / qps)
    except Exception as error:
      # 获取整体报错信息
      print(error)

# 查询人脸数据库列表
def list_face_dbs():
    list_face_dbs_request = ListFaceDbsRequest()
    try:
        runtime_option = RuntimeOptions()
        response = client.list_face_dbs_with_options(list_face_dbs_request, runtime_option)
        time.sleep(1.0 / qps)
        # 获取整体结果
        return response.body.to_map()['Data']['DbList']
    except Exception as error:
        # 获取整体报错信息
        print(error)

# 增加人脸样本
def add_face_entity(dbName, entityId, labels=None):
    add_face_entity_request = AddFaceEntityRequest(db_name=dbName,
                                                   entity_id=entityId,
                                                   labels=labels)
    try:
        runtime_option = RuntimeOptions()
        response = client.add_face_entity_with_options(add_face_entity_request, runtime_option)
        time.sleep(1.0 / qps)
    except Exception as error:
        # 获取整体报错信息
        print(error)

# 查询人脸样本
def get_face_entity(dbName, entityId):
    get_face_entity_request = GetFaceEntityRequest(db_name=dbName,
                                                   entity_id=entityId)
    try:
        runtime_option = RuntimeOptions()
        response = client.get_face_entity_with_options(get_face_entity_request, runtime_option)
        time.sleep(1.0 / qps)
        # 获取整体结果
        return response.body.to_map()['Data']
    except Exception as error:
        # 获取整体报错信息
        print(error)

# 查询人脸样本列表
def list_face_entities(dbName):
    list_face_entities_request = ListFaceEntitiesRequest(db_name=dbName)
    try:
        runtime_option = RuntimeOptions()
        response = client.list_face_entities_with_options(list_face_entities_request, runtime_option)
        time.sleep(1.0 / qps)
        # 获取整体结果
        return response.body.to_map()['Data']['Entities']
    except Exception as error:
        # 获取整体报错信息
        print(error)

# 更新人脸样本

# 添加人脸数据
def add_face(dbName, entityId, face_path, extraData,
             quality_score_threshold=50,
             similarity_score_threshold_in_entity=50,
             similarity_score_threshold_between_entity=50):
    stream = open(face_path, 'rb')
    add_face_request = AddFaceAdvanceRequest(image_url_object=stream,
                                             db_name=dbName,
                                             entity_id=entityId,
                                             extra_data=extraData,
                                             quality_score_threshold=quality_score_threshold,
                                             similarity_score_threshold_in_entity=similarity_score_threshold_in_entity,
                                             similarity_score_threshold_between_entity=similarity_score_threshold_between_entity)
    try:
        runtime_option = RuntimeOptions()
        response = client.add_face_advance(add_face_request, runtime_option)
        time.sleep(1.0 / qps)
    except Exception as error:
        # 获取整体报错信息
        print(error)
    finally:
        stream.close()

# 批量添加人脸数据
def batch_add_face(dbName, entityId, face_paths,
                   quality_score_threshold=50,
                   similarity_score_threshold_in_entity=50,
                   similarity_score_threshold_between_entity=50):
    streams = []
    faces = []
    for i, face_path in enumerate(face_paths):
        streams.append(open(face_path, 'rb'))
        faces.append(BatchAddFacesAdvanceRequestFaces(image_urlobject=streams[-1],
                                                      extra_data=face_path))
    batch_add_face_request = BatchAddFacesAdvanceRequest(db_name=dbName, entity_id=entityId, faces=faces,
                                                         quality_score_threshold=quality_score_threshold,
                                                         similarity_score_threshold_in_entity=similarity_score_threshold_in_entity,
                                                         similarity_score_threshold_between_entity=similarity_score_threshold_between_entity)

    try:
        runtime_option = RuntimeOptions()
        response = client.batch_add_faces_advance(batch_add_face_request, runtime_option)
        time.sleep(1.0 / qps)
    except Exception as error:
        # 获取整体报错信息
        print(error)
    finally:
        for stream in streams:
            stream.close()

# 删除人脸
def delete_face(dbName, faceId):
    delete_face_request = DeleteFaceRequest(db_name=dbName, face_id=faceId)
    try:
        runtime_option = RuntimeOptions()
        response = client.delete_face_with_options(delete_face_request, runtime_option)
        time.sleep(1.0 / qps)
    except Exception as error:
        # 获取整体报错信息
        print(error)

# 删除人脸样本
def delete_face_entity(dbName, entityId):
    delete_face_entity_request = DeleteFaceEntityRequest(db_name=dbName, entity_id=entityId)
    try:
        runtime_option = RuntimeOptions()
        response = client.delete_face_entity_with_options(delete_face_entity_request, runtime_option)
        time.sleep(1.0 / qps)
    except Exception as error:
        # 获取整体报错信息
        print(error)

# 删除人脸数据库
def delete_face_db(dbName):
    delete_face_db_request = DeleteFaceDbRequest(name=dbName)
    try:
        runtime_option = RuntimeOptions()
        response = client.delete_face_db_with_options(delete_face_db_request, runtime_option)
        time.sleep(1.0 / qps)
    except Exception as error:
        # 获取整体报错信息
        print(error)

def clear_db(dbName):
    entities = list_face_entities(dbName)
    for entity in tqdm(entities,total=len(entities)):
        delete_face_entity(dbName, entity['EntityId'])
    delete_face_db(dbName)

def upload_db(dbName, db_path):
    create_face_db(dbName)
    folder = os.path.expanduser(db_path)
    class_names = os.listdir(folder)
    for i, class_name in tqdm(enumerate(class_names), total=len(class_names)):
        class_dir = os.path.join(folder, class_name)
        if os.path.isdir(class_dir):
            add_face_entity(dbName, i, class_name)
            image_names = os.listdir(class_dir)
            image_paths = [os.path.join(class_dir, img_name) for img_name in image_names]
            for j in range(0, len(image_paths), 5):
                chunk_paths = image_paths[j:min(j + 5, len(image_paths))]
                batch_add_face(dbName, i, chunk_paths)

def json_print(obj):
    print(json.dumps(obj, indent=2))

if __name__ == '__main__':
    # create_face_db("test")
    # result = list_face_dbs()
    # add_face_entity('test', 3)
    # result = get_face_entity('test', 3)
    # result = list_face_entities('test')
    # add_face('test',3,r'D:/yy/source/Desktop/读研是一条艰苦的道路/1. 论文/做实验/myCode/4 FrAdv/AdvFaceGAN/test/fake.png','fake_Aaron_Peirsol_0003.jpg')
    # batch_add_face('test', 3, [
    #     r'D:\datasets\debug\Aaron_Peirsol\Aaron_Peirsol_0002.jpg',
    #     r'D:\datasets\debug\Aaron_Peirsol\Aaron_Peirsol_0003.jpg',
    #     r'D:\datasets\debug\Aaron_Peirsol\Aaron_Peirsol_0004.jpg'
    # ])
    # delete_face('test', 163929891)
    # delete_face_entity('test',3)
    # delete_face_db('test')
    # clear_db('test')
    # upload_db('test', r'D:\datasets\debug')

    result = face_compare(face1_path=r"D:/yy/source/Desktop/读研是一条艰苦的道路/1. 论文/做实验/myCode/4 FrAdv/AdvFaceGAN/test/fake.png",
                          face2_path=r"D:\datasets\debug\Aaron_Peirsol\Aaron_Peirsol_0004.jpg")

    # result = search_face(face_path=r"C:\Users\28769\Pictures\Camera Roll\img1.jpg", dbName='test')
    #
    json_print(result)
    pass