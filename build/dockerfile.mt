# docker build -t pawlyglot/mt:1.0.0 -f build/dockerfile.mt .
FROM pawlyglot/base:1.0.0

COPY build/requirements.mt.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.mt.txt

WORKDIR /mt

# setup protobuf
RUN mkdir ./proto
COPY proto/mt.proto ./proto/mt.proto

COPY modules/machine_translation/src ./src
RUN python -m grpc_tools.protoc -I ./proto --python_out=./src --pyi_out=./src --grpc_python_out=./src ./proto/mt.proto

CMD ["python3", "src/serve.py"]
