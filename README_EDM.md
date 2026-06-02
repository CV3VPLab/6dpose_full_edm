# 6D Pose EDM/TRT Notes

This branch keeps the professor baseline pipeline structure and replaces the
Step5 matcher path with DINO TRT + EDM TRT.

Pipeline:

```text
Step1 YOLO/SAM2
-> DINOv2 retrieval
-> EDM matching
-> Step6 PnP initial pose
-> optional Step7 Render&Compare refinement
```

## Files Not Included In Git

Large/local assets are intentionally ignored:

```text
data/
mtdew/
md_query/
sam2_repo/
weights/*.pt
weights/*.onnx
```

Expected local files:

```text
weights/best_3cls_v2.pt
weights/sam2.1_hiera_tiny.pt
weights/dino_vits14_224.onnx
weights/edm_outdoor_w672_h672_topk2469.onnx
```

The EDM ONNX can be regenerated with:

```bash
git clone https://github.com/chicleee/EDM.git /path/to/EDM_repo
python scripts/export_edm_pair_onnx.py \
  --edm_repo /path/to/EDM_repo \
  --ckpt /path/to/EDM_repo/weights/edm_weights/edm_outdoor.ckpt \
  --height 672 \
  --width 672 \
  --out weights/edm_outdoor_w672_h672_topk2469.onnx
```

## Runtime Libraries

`EDM_TRT` and `DINO_TRT` use ONNXRuntime TensorRT EP. The runtime linker must
find TensorRT and cuDNN libraries, otherwise ONNXRuntime falls back to CPU or
fails with errors such as:

```text
libnvinfer.so.10: cannot open shared object file
libcudnn.so.8: cannot open shared object file
```

For a fresh environment, use the integrated setup:

```bash
bash setup_full_edm.sh
```

This installs the baseline dependencies plus ONNXRuntime/TensorRT runtime
dependencies. It does not clone EDM and does not download private/local weights
or datasets. EDM source is only needed if you want to regenerate the ONNX.

Before running:

```bash
conda activate 6dpose_test
source setup_edm_env.sh
python estimate_object_pose.py
```

If auto-detection fails, set these manually:

```bash
export TENSORRT_LIB_DIR=/path/to/tensorrt_libs
export CUDNN_LIB_DIR=/path/to/cudnn8/lib
export LD_LIBRARY_PATH="$TENSORRT_LIB_DIR:$CUDNN_LIB_DIR:$LD_LIBRARY_PATH"
```

Successful logs should include:

```text
[DINO TRT] Providers: ['TensorrtExecutionProvider', ...]
[EDM TRT] Providers: ['TensorrtExecutionProvider', ...]
```

## Benchmark Flag

`ope_config.json` has:

```json
"skip_refine": true
```

This is not part of the professor baseline. It is a benchmark flag added to
measure Step5/6 latency only. When true, Step7 Render&Compare is skipped and
the Step6 PnP pose is returned. `tracking loss` is therefore `nan`.

Use this for full pipeline behavior:

```json
"skip_refine": false
```
