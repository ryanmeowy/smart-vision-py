import subprocess
import sys

command = [
    sys.executable,
    "-m",
    "grpc_tools.protoc",
    "-I./protos",
    "--python_out=.",
    "--grpc_python_out=.",
    "./protos/vision.proto",
]

print("Executing:", " ".join(command))
completed = subprocess.run(command, check=False)
if completed.returncode != 0:
    raise SystemExit(completed.returncode)
print("✅ gRPC code generated successfully!")
