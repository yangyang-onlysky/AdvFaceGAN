
import json
import os
import time

# import official_api.FacePPSDK
from official_api.FacePPSDK.facepp import API,File
# 导入图片处理类
import official_api.FacePPSDK.ImagePro

def json_print(obj):
    print(json.dumps(obj, indent=2))

# get API Key API Secret from environment variable
API_KEY = os.environ.get('FACEPP_API_KEY')
API_SECRET = os.environ.get('FACEPP_API_SECRET')
api = API(API_KEY, API_SECRET)

qps = 10

def face_compare(face1_path, face2_path):
    time.sleep(1.0 / qps)
    confidence = None
    try:
        resp = api.compare(image_file1=File(face1_path),
                           image_file2=File(face2_path))
        confidence = resp['confidence']
    except Exception as error:
        # 获取整体报错信息
        print("error:", error)
    return confidence

if __name__ == '__main__':
    # -----------------------------------------------------------人脸识别部分-------------------------------------------

    # 人脸检测：https://console.faceplusplus.com.cn/documents/4888373
    # res = api.detect(image_url=detech_img_url, return_attributes="gender,age,smiling,headpose,facequality,"
    #                                                        "blur,eyestatus,emotion,ethnicity,beauty,"
    #                                                        "mouthstatus,skinstatus")
    # print_result(printFuctionTitle("人脸检测"), res)

    # 人脸比对：https://console.faceplusplus.com.cn/documents/4887586
    # compare_res = face_compare(face1_path="../testtest/fake.png",
    #                            face2_path="../testtest/target.png")
    # json_print(compare_res)

    # 人脸搜索：https://console.faceplusplus.com.cn/documents/4888381
    # 人脸搜索步骤
    # 1,创建faceSet:用于存储人脸信息(face_token)
    # 2,向faceSet中添加人脸信息(face_token)
    # 3，开始搜索

    # 删除无用的人脸库，这里删除了，如果在项目中请注意是否要删除
    # api.faceset.delete(outer_id='faceplusplus', check_empty=0)
    # # 1.创建一个faceSet
    # ret = api.faceset.create(outer_id='faceplusplus')
    #
    # # 2.向faceSet中添加人脸信息(face_token)
    # faceResStr=""
    # res = api.detect(image_file=File(faceSet_img))
    # faceList = res["faces"]
    # for index in range(len(faceList)):
    #     if(index==0):
    #         faceResStr = faceResStr + faceList[index]["face_token"]
    #     else:
    #         faceResStr = faceResStr + ","+faceList[index]["face_token"]
    #
    # api.faceset.addface(outer_id='faceplusplus', face_tokens=faceResStr)
    #
    # # 3.开始搜索相似脸人脸信息
    # search_result = api.search(image_file=File(face_search_img), outer_id='faceplusplus')
    # print_result('search', search_result)

    # -----------------------------------------------------------人体识别部分-------------------------------------------

    # 人体抠像:https://console.faceplusplus.com.cn/documents/10071567
    # segment_res = api.segment(image_file=File(segment_img))
    # f = open('./imgResource/demo-segment.b64', 'w')
    # f.write(segment_res["result"])
    # f.close()
    # print_result("segment", segment_res)
    # # 开始抠像
    # PythonSDK.ImagePro.ImageProCls.getSegmentImg("./imgResource/demo-segment.b64")

    # -----------------------------------------------------------证件识别部分-------------------------------------------
    # 身份证识别:https://console.faceplusplus.com.cn/documents/5671702
    # ocrIDCard_res = api.ocridcard(image_url="https://gss0.bdstatic.com/94o3dSag_xI4khGkpoWK1HF6hhy/baike/"
    #                                         "c0%3Dbaike80%2C5%2C5%2C80%2C26/sign=7a16a1be19178a82da3177f2976a18e8"
    #                                         "/902397dda144ad34a1b2dcf5d7a20cf431ad85b7.jpg")
    # print_result('ocrIDCard', ocrIDCard_res)

    # 银行卡识别:https://console.faceplusplus.com.cn/documents/10069553
    # ocrBankCard_res = api.ocrbankcard(image_url="http://pic.5tu.cn/uploads/allimg/1107/191634534200.jpg")
    # print_result('ocrBankCard', ocrBankCard_res)

    # -----------------------------------------------------------图像识别部分-------------------------------------------
    # 人脸融合：https://console.faceplusplus.com.cn/documents/20813963
    # template_rectangle参数中的数据要通过人脸检测api来获取
    # mergeFace_res = api.mergeface(template_file=File(segment_img), merge_file=File(merge_img),
    #                               template_rectangle="130,180,172,172")
    # print_result("mergeFace", mergeFace_res)
    #
    # # 开始融合
    # PythonSDK.ImagePro.ImageProCls.getMergeImg(mergeFace_res["result"])
    pass
