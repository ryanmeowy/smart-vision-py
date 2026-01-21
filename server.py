import time
from concurrent import futures

import grpc

import vision_pb2
import vision_pb2_grpc
from core.caption_service import caption_service
from core.embedding_service import embedding_service
from core.ocr_service import ocr_service


class VisionServer(vision_pb2_grpc.VisionServiceServicer):

    def EmbedText(self, request, context):
        try:
            print(f"📝 Request EmbedText: {request.text}")
            vector = embedding_service.embed_text(request.text)
            return vision_pb2.EmbeddingResponse(vector=vector[0].tolist(), dim=vector[0].size)
        except Exception as e:
            print(f"Error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vision_pb2.EmbeddingResponse()

    def EmbedImage(self, request, context):
        try:
            print(f"🖼️ Request EmbedImage: {request.url}")
            vector = embedding_service.embed_image(request.url)
            return vision_pb2.EmbeddingResponse(vector=vector[0].tolist(), dim=vector[0].size)
        except Exception as e:
            print(f"Error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vision_pb2.EmbeddingResponse()

    def GenerateFileName(self, request, context):
        try:
            print(f"🔍 Request gen file name: {request.image_url}")
            name = caption_service.generate_name(request.image_url)
            return vision_pb2.GenFileNameResponse(name=name)
        except Exception as e:
            print(f"Error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vision_pb2.GenFileNameResponse()

    def GenerateTags(self, request, context):
        try:
            print(f"🔍 Request gen tag: {request.image_url}")
            name = caption_service.generate_tags(request.image_url)
            return vision_pb2.GenTagsResponse(tag=name)
        except Exception as e:
            print(f"Error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vision_pb2.GenTagsResponse()

    def ExtractText(self, request, context):
        try:
            print(f"🔍 Request OCR: {request.image_url}")
            result = ocr_service.extract_text(request.image_url)
            return vision_pb2.OcrResponse(full_text=result[0], lines=result[1])
        except Exception as e:
            print(f"Error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vision_pb2.OcrResponse()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vision_pb2_grpc.add_VisionServiceServicer_to_server(VisionServer(), server)
    port = '[::]:50051'
    server.add_insecure_port(port)
    print(f"✅ gRPC Server started on {port}")
    server.start()
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
