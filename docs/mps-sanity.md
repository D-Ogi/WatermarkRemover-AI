# Apple MPS sanity check

This is a device-agreement check, not a benchmark. The repository has no labeled
watermark dataset and no fixed public clip with reference masks or scores. The
runs below compare CPU and MPS on the same inputs. They do not measure removal
quality against ground truth, and they do not represent other Apple, NVIDIA, or
CPU machines.

Maintainers still need to decide what a benchmark should require. This note does
not define that set, metric, or acceptance bar.

## Environment

| Item | Value |
| --- | --- |
| Machine | Apple Silicon, MPS available, CUDA unavailable |
| PyTorch | 2.14.1 |
| Fallback | `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| LaMA | cached `big-lama.pt`, SHA-256 `344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9` |
| Florence-2 | `florence-community/Florence-2-large` @ `4271c66b88cdbc05735372ec13b2360108de5317` |
| CLI defaults | detection prompt `watermark`, detection skip 1, max bbox 10%, no fade, no transparency |

The source video and both outputs stayed on the local machine. They are not in
this pull request.

## Synthetic inputs

The images in this section are synthetic. The weights are the cached LaMA file
and the pinned Florence-2 revision in the table above, not a generated model.

LaMA inpainted a 128×160 random image with a 50×70 mask. CPU and MPS both
returned finite images of the same shape, left unmasked pixels unchanged, and
changed the masked pixels. Their maximum absolute difference was 1, and the mean
absolute difference was 0.0001.

Florence-2 detected the same boxes on a 256×256 image on both devices: the full
frame `[0, 0, 255, 255]`. That only shows the two devices agreed. It does not
show that the detection was correct.

## Video comparison

Input: local H.264 clip, 720×1280, 24 fps, 361 frames, 15.04 seconds. Each run
used the command-line defaults above.

```sh
python remwm.py INPUT videos/compare/cpu.mp4 --device cpu
python remwm.py INPUT videos/compare/mps.mp4 --device mps
```

| Device | Wall time | Frame loop | Output |
| --- | --- | --- | --- |
| CPU | 815.7 s | 13 min 29 s, about 2.24 s/frame | 361 frames, 720×1280, 24 fps, 12,042,738 bytes |
| MPS | 229.0 s | 3 min 41 s, about 0.61 s/frame | 361 frames, 720×1280, 24 fps, 12,042,766 bytes |

Decoded frames from the two outputs:

| Comparison | Result |
| --- | --- |
| Frames read | 361/361, matching width, height, and frame rate |
| Mean absolute difference | 0.000132 |
| PSNR | 83.25 dB |
| Frames with any difference | 43 |
| Frames whose largest pixel difference is above 5 | 21 |
| Largest pixel difference | 14, on frame 10 |
| Mean absolute difference of that frame | 0.0057 |

MPS finished the same clip about 3.6 times faster than CPU. The pixel
differences are small, so this clip did not show a visible device split. A
benchmark would still need a fixed public set, reference masks or expected
output, and a quality metric. None of those were part of this check.
