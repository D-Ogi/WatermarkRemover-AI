# Third-party notices

## LaMA processing adapter

The crop, symmetric padding, tensor normalization and compositing behavior in
`lama_inpaint` is adapted from [Sanster/IOPaint](https://github.com/Sanster/IOPaint),
revision `61a759fb3f332bacdce8b2813f4837495c9b86e0`, specifically
`iopaint/model/lama.py`, `iopaint/model/base.py` and `iopaint/helper.py`.
The upstream Apache-2.0 license is reproduced in
[licenses/IOPaint-Apache-2.0.txt](licenses/IOPaint-Apache-2.0.txt).

Changes: standalone RGB prediction/BGR output adapter; input validation;
non-mutating compositing; empty-mask bypass; soft-mask crop selection;
verified atomic model downloads; removal of the IOPaint registry, schemas,
CLI, diffusion models and their runtime dependencies.

## Model weights

Weights are downloaded separately from the existing upstream
[big-lama.pt release](https://github.com/Sanster/models/releases/tag/add_big_lama).
They are not included in this repository. The artifact is 205669692 bytes;
SHA-256: `344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9`.
This digest was calculated from the artifact matching IOPaint's pinned MD5
`e3aa4aaa15225a33ec84f9f4bc47e500`; the legacy release has no GitHub SHA-256
asset digest. The runtime uses the pinned SHA-256, not MD5 or an environment
variable supplied checksum. This is an integrity pin, not an upstream signature.
For model provenance and terms, see [LaMA](https://github.com/advimman/lama)
and the artifact publisher. Code licensing does not replace model/data terms.
