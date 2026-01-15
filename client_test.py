import os
import sys

import grpc

sys.path.append(os.path.abspath("protos"))
import vision_pb2 as pb2
import vision_pb2_grpc as pb2_grpc

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = pb2_grpc.VisionServiceStub(channel)

    # resp = stub.ExtractText(pb2.GenRequest(image_url = "https://images.pexels.com/photos/6202185/pexels-photo-6202185.jpeg"))
    resp = stub.ExtractText(pb2.GenRequest(image_url = "https://images.pexels.com/photos/6202185/pexels-photo-6202185.jpeg", prompt = "请精确提取图中的所有文本内容，包括印刷体和清晰的手写体。请忽略水印，并丢弃无意义的文本（如单个标点符号、无上下文的孤立字符）。若图中没有文本、文本无法识别或难以识别，请输出“-1”。若有文本，请直接输出提取到的文本，不要输出任何与图中文本无关的内容。"))
    # print(resp)

    # resp = stub.GenerateFileName(pb2.GenRequest(image_url = "https://images.pexels.com/photos/8976496/pexels-photo-8976496.jpeg", prompt = "为所附图片生成一个3-6字的中文图片名，要求简洁、达意、富有美感，直接输出名称即可。"))

    # resp = stub.GenerateTags(pb2.GenRequest(image_url = "https://images.pexels.com/photos/8976496/pexels-photo-8976496.jpeg", prompt = """请分析这张图片，提取 3-5 个核心标签，包含物体、场景、风格。 请直接返回一个 JSON 字符串数组，不要包含 Markdown 格式或其他废话。例如：["风景", "雪山", "日落"]"""))
    print(resp)

if __name__ == '__main__':
    run()