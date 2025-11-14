docker run -d \
  --name gpustack_new \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  --restart unless-stopped \
  -p 9101:9880 \
  -p 9990:9881 \
  -v /var/lib/gpustack:/var/lib/gpustack \
  -v /etc/gpustack:/etc/gpustack \
  -v /var/log/gpustack:/var/log/gpustack \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -e GPUSTACK_HOST=0.0.0.0 \
  -e GPUSTACK_PORT=9880 \
  -e ASCEND_VISIBLE_DEVICES=0 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  gpustack/gpustack:main-npu