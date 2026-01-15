import grpc
from concurrent import futures
import time

# 导入生成的代码
import vision_pb2
import vision_pb2_grpc

# 导入业务逻辑
from core.embedding_service import embedding_service
from utils.image_loader import load_image_from_url
from core.caption_service import caption_service
from core.ocr_service import ocr_service


class VisionServer(vision_pb2_grpc.VisionServiceServicer):

    def EmbedText(self, request, context):
        try:
            print(f"📝 Request EmbedText: {request.text}")
            # vector = caption_service.get_embedding(request.text, None)
            vector = embedding_service.embed_text(request.text)
            return vision_pb2.EmbeddingResponse(vector=vector, dim=len(vector))
        except Exception as e:
            print(f"Error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vision_pb2.EmbeddingResponse()

    def EmbedImage(self, request, context):
        try:
            print(f"🖼️ Request EmbedImage: {request.url}")
            # 1. 下载图片
            image = load_image_from_url(request.url)
            # 2. 计算向量
            # vector = caption_service.get_embedding(None, image)
            vector = embedding_service.embed_image(image)
            return vision_pb2.EmbeddingResponse(vector=vector, dim=len(vector))
        except Exception as e:
            print(f"Error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vision_pb2.EmbeddingResponse()

    def ExtractText(self, request, context):
        try:
            print(f"🔍 Request OCR: {request.image_url}")
            image = load_image_from_url(request.image_url)
            prompt = request.prompt if request.prompt else "请精确提取图中的所有文本内容，包括印刷体和清晰的手写体。请忽略水印，并丢弃无意义的文本（如单个标点符号、无上下文的孤立字符）。若图中没有文本、文本无法识别或难以识别，请输出“-1”。若有文本，请直接输出提取到的文本，不要输出任何与图中文本无关的内容。"
            full_text = caption_service.generate_text(image, prompt)
            lines = full_text.split('\n')
            lines = [line.strip() for line in lines if line.strip()]
            return vision_pb2.OcrResponse(full_text=full_text, lines=lines)
        except Exception as e:
            print(f"Error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vision_pb2.OcrResponse()

    def GenerateCaption(self, request, context):
        """
        流式生成接口
        request.image_url: 图片链接
        request.prompt: (可选) 比如 "写一个小红书文案"
        """
        try:
            print(f"✨ Request Gen: {request.image_url}")

            # 如果 Java 端没传 prompt，给个默认值
            prompt = request.prompt if request.prompt else "请详细描述这张图片"

            # 调用流式生成
            for chunk in caption_service.stream_generate(request.image_url, prompt):
                # 实时返回给 Java
                yield vision_pb2.StringResponse(content=chunk)

        except Exception as e:
            print(f"Error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            # 流式接口出错也得 yield 一个空或者错误信息，防止客户端卡死
            yield vision_pb2.StringResponse(content=f"[Error: {str(e)}]")

    def GenerateFileName(self, request, context):
        try:
            print(f"🔍 Request gen file name: {request.image_url}")
            image = load_image_from_url(request.image_url)
            prompt = request.prompt if request.prompt else "为所附图片生成一个3-6字的中文图片名，要求简洁、达意、富有美感，直接输出名称即可。"
            name = caption_service.generate_text(image, prompt)
            return vision_pb2.GenFileNameResponse(name=name)
        except Exception as e:
            print(f"Error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vision_pb2.GenFileNameResponse()

    def GenerateTags(self, request, context):
        try:
            print(f"🔍 Request gen tag: {request.image_url}")
            image = load_image_from_url(request.image_url)
            prompt = request.prompt if request.prompt else """请分析这张图片，提取 3-5 个核心标签，包含物体、场景、风格。 请直接返回一个 JSON 字符串数组，不要包含 Markdown 格式或其他废话。例如：["风景", "雪山", "日落"]"""
            name = caption_service.generate_text_list(image, prompt)
            return vision_pb2.GenTagsResponse(tag=name)
        except Exception as e:
            print(f"Error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vision_pb2.GenTagsResponse()

    # def ExtractText(self, request, context):
    #     try:
    #         print(f"🔍 Request OCR: {request.image_url}")
    #         image = load_image_from_url(request.image_url)
    #         result = ocr_service.extract_text(image)
    #         return vision_pb2.OcrResponse(full_text=result[0], lines=result[1])
    #     except Exception as e:
    #         print(f"Error: {e}")
    #         context.set_code(grpc.StatusCode.INTERNAL)
    #         context.set_details(str(e))
    #         return vision_pb2.OcrResponse()


def serve():
    # 创建 gRPC 服务器，使用线程池 (最大10并发)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # 注册服务
    vision_pb2_grpc.add_VisionServiceServicer_to_server(VisionServer(), server)

    # 监听端口
    port = '[::]:50051'
    server.add_insecure_port(port)
    print(f"✅ gRPC Server started on {port}")
    print("   - Chinese-CLIP (768 dim)")
    print("   - PaddleOCR (v4)")

    server.start()

    # 保持运行
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
