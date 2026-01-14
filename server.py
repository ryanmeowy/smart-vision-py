import grpc
from concurrent import futures
import time

# 导入生成的代码
import vision_pb2
import vision_pb2_grpc

# 导入业务逻辑
from core.embedding_service import embedding_service
from core.ocr_service import ocr_service
from utils.image_loader import load_image_from_url
from core.caption_service import caption_service



class VisionServer(vision_pb2_grpc.VisionServiceServicer):

    def EmbedText(self, request, context):
        try:
            print(f"📝 Request EmbedText: {request.text}")
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
            vector = embedding_service.embed_image(image)
            return vision_pb2.EmbeddingResponse(vector=vector, dim=len(vector))
        except Exception as e:
            print(f"Error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vision_pb2.EmbeddingResponse()

    def ExtractText(self, request, context):
        try:
            print(f"🔍 Request OCR: {request.url}")
            image = load_image_from_url(request.url)
            full_text = caption_service.extract_text_ocr(image)
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