# docker build -t pawlyglot/mt:1.0.0-dev -f dockerfile.dev.mt .
FROM pawlyglot/base:1.0.0

WORKDIR /mt

COPY requirements.mt.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.mt.txt

CMD ["sh", "-c", "python3 -m grpc_tools.protoc -I ./src --python_out=./src --pyi_out=./src --grpc_python_out=./src ./src/mt.proto && python3 src/serve.py"]
