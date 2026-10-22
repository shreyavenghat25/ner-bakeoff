| Model | Size / quantisation | F1 | F1 PERSON | F1 ORG | F1 LOCATION | F1 EVENT | Median latency (s) | p95 latency (s) | Peak memory (MB) | Cost / 1k snippets (USD) | Cost assumption |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gliner-gliner_medium-v2.1 | 195M params, fp32 (CPU) | 0.800 | 0.772 | 0.780 | 0.943 | 0.500 | 0.060 | 0.064 | 1518 | 0.0030 | EC2 c7i.xlarge (4 vCPU, 8 GiB): CPU inference for a ~200M-parameter model @ $0.1785/h (https://www.doit.com/compute/spot/us-east-1/c7i.xlarge) |
| ollama-qwen3-4b | 4.0B, Q4_K_M | 0.953 | 0.941 | 0.980 | 0.963 | 0.903 | 1.966 | 3.475 | 3048 | 0.4606 | EC2 g6.xlarge (4 vCPU, 16 GiB, 1x NVIDIA L4 24 GB): smallest current NVIDIA GPU instance; a 4B Q4 model fits easily @ $0.8048/h (https://www.doit.com/compute/spot/us-east-1/g6.xlarge) |
| spacy-en_core_web_sm | en_core_web_sm v3.8.0, fp32 (CPU) | 0.644 | 0.727 | 0.621 | 0.658 | 0.421 | 0.003 | 0.004 | 308 | 0.0001 | EC2 c7i.large (2 vCPU, 4 GiB): spaCy sm needs well under 1 GB @ $0.0892/h (https://www.doit.com/compute/spot/us-east-1/c7i.large) |

Hardware: Apple M2, 8 logical cores, 8.0 GB RAM, GPU: none detected, Darwin 23.4.0, Python 3.11.17.
F1 here is from the local scorer; confirm every number with the official score.py.
