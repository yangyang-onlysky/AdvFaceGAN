# -*- coding: utf-8 -*-
import base64
import os
import json
import time

from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.iai.v20200303 import iai_client, models

# get API Key API Secret from environment variable
cred = credential.Credential(os.environ.get("TENCENTCLOUD_SECRET_ID"),
                             os.environ.get("TENCENTCLOUD_SECRET_KEY"))
httpProfile = HttpProfile()
httpProfile.endpoint = "iai.ap-chengdu.tencentcloudapi.com"
clientProfile = ClientProfile()
clientProfile.httpProfile = httpProfile
client = iai_client.IaiClient(cred, "ap-chengdu", clientProfile)

qps = 50

def image_to_base64(image_path):
    # 打开图片文件
    with open(image_path, "rb") as image_file:
        # 读取文件内容
        image_data = image_file.read()
        # 将文件内容编码为Base64
        base64_encoded = base64.b64encode(image_data).decode('utf-8')
    return base64_encoded


def face_compare(face1_path, face2_path):
    try:
        # 实例化一个请求对象,每个接口都会对应一个request对象
        req = models.CompareFaceRequest()
        params = {
            "ImageA": image_to_base64(face1_path),
            "ImageB": image_to_base64(face2_path),
        }
        req.from_json_string(json.dumps(params))

        time.sleep(1.0/qps)
        # 返回的resp是一个CompareFaceResponse的实例，与请求对象对应
        resp = client.CompareFace(req)
        # 输出json格式的字符串回包
        return resp.Score

    except TencentCloudSDKException as err:
        print(err)
    return None

if __name__ == '__main__':
    result = face_compare(face1_path=r"D:\datasets\debug\Aaron_Peirsol\Aaron_Peirsol_0002.jpg",
                          face2_path=r"D:\datasets\debug\Aaron_Peirsol\Aaron_Peirsol_0003.jpg")
    print(result)
