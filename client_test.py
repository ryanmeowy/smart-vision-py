import os
import sys

import grpc

sys.path.append(os.path.abspath("protos"))
import vision_pb2 as pb2
import vision_pb2_grpc as pb2_grpc

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = pb2_grpc.VisionServiceStub(channel)

    resp = stub.ExtractText(pb2.GenRequest(image_url = "https://images.pexels.com/photos/1148572/pexels-photo-1148572.jpeg"))
    # print(resp)

    # resp = stub.GenerateFileName(pb2.GenRequest(image_url = "https://images.pexels.com/photos/8976496/pexels-photo-8976496.jpeg", prompt = "为所附图片生成一个3-6字的中文图片名，要求简洁、达意、富有美感，直接输出名称即可。"))

    # resp = stub.GenerateTags(pb2.GenRequest(image_url = "https://images.pexels.com/photos/8976496/pexels-photo-8976496.jpeg", prompt = """请分析这张图片，提取 3-5 个核心标签，包含物体、场景、风格。 请直接返回一个 JSON 字符串数组，不要包含 Markdown 格式或其他废话。例如：["风景", "雪山", "日落"]"""))
    print(resp)

if __name__ == '__main__':
    run()