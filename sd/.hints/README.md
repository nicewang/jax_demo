```bash
curl -LO https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-darwin-amd64-v3.5.1.zip
unzip git-lfs-darwin-amd64-v3.5.1.zip
cd git-lfs-3.5.
sudo ./install.sh

GIT_LFS_SKIP_SMUDGE=1 git submodule add https://huggingface.co/NiceWang/sd-naruto-tpu-float32 sd/float32/model_naruto

git rm -f sd/float32/model_naruto_weights

git remote set-url origin https://huggingface.co/NiceWang/sd-naruto-tpu

git add .gitattributes pic/Sasuke.png

git config --global http.version HTTP/1.1

git reset --soft origin/main

git rm --cached pic/Sasuke.png

git lfs track "*.png"

export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
```
- stable-diffusion-v1-5/stable-diffusion-v1-5