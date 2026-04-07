## How to build docker file

### conda activate python312

### Using He Liu's docker image

### find out the correct path and correct directory

### GIT_LFS_SKIP_SMUDGE=1 uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple

### GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

### uv pip install datasets==3.0.0

### cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/

```code
rm -rf .venv
export UV_LINK_MODE=copy
GIT_LFS_SKIP_SMUDGE=1 uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
source .venv/bin/activate
uv pip install datasets==3.0.0
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```