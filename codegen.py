import os

command = (
    "python -m grpc_tools.protoc "
    "-I./protos "
    "--python_out=. "
    "--grpc_python_out=. "
    "./protos/vision.proto"
)

print(f"Executing: {command}")
os.system(command)
print("✅ gRPC code generated successfully!")