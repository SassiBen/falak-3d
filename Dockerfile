FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY constraints.txt /opt/constraints.txt
# L'index cu121 reste prioritaire : c'est le dernier à publier des roues torch
# avec des noyaux sm_61. PyPI sert d'appoint pour une version de cuDNN que
# l'index PyTorch ne publie plus.
RUN pip install --no-cache-dir --retries 10 --timeout 120 -c /opt/constraints.txt \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121 \
        --extra-index-url https://pypi.org/simple \
    && pip install --no-cache-dir -c /opt/constraints.txt \
        "git+https://github.com/openai/shap-e.git@8625e7c15526d8510a2292f92165979268d0e945" \
        pyyaml ipywidgets

# Échoue au build plutôt qu'au premier job si torch a perdu Pascal en route.
RUN python -c "import torch; archs = torch.cuda.get_arch_list(); \
    assert any(a.startswith('sm_6') for a in archs), archs"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -c /opt/constraints.txt .
COPY presets ./presets

ENV FALAK3D_DATA_DIR=/data
VOLUME /data
EXPOSE 8001
CMD ["falak3d", "serve"]
