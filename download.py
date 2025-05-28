from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="wonjik-kim/MultiClassSmoothedHingeLoss", filename="resnet50_MCSH_m7.pth", local_dir="./pretrained")
hf_hub_download(repo_id="wonjik-kim/MultiClassSmoothedHingeLoss", filename="ViT_base_MCSH_m7.pth", local_dir="./pretrained")