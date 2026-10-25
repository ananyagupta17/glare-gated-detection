# Flare7K++ restoration: licence and checkpoint provenance

Checked 2026-10-20 against the upstream repository (github.com/ykdai/Flare7K, commit `d1fb66e`,
2024-08-30): `LICENSE` and the README's *License* and *Citation* sections.

## Licence

| | |
|---|---|
| Licence | **S-Lab License 1.0** (Copyright 2022 S-Lab, NTU). A BSD-3-style text restricted to non-commercial use. |
| Covers | The code, and per the README "the dataset and code". The pretrained checkpoint is distributed by the same authors under the same terms. |
| Research use | **Allowed.** Use and redistribution "for non-commercial purpose", with or without modification. A university capstone and an academic paper are non-commercial. |
| Commercial use | Not granted. It requires contacting the authors. |
| Conditions | (1) Source redistributions keep the copyright notice and licence text. (2) Binary redistributions, which includes sharing the checkpoint, reproduce the notice in the accompanying documentation. (3) S-Lab's or the contributors' names may not be used to endorse or promote derived work. |
| Attribution | The licence itself requires the notice to be kept, not a citation. The README asks users to cite the work. Treat that as required for the paper. |
| Warranty | None ("AS IS"). |

**In this repo:** `THIRD_PARTY_LICENSES/Flare7K-LICENSE.txt`, from Vihaan's handover, is included on
this branch and meets the notice conditions (1 and 2).
The checkpoint is shared outside git. Whoever passes it on should pass the licence file
with it.

**Cite in the paper** (from the Flare7K README):

- Dai, Li, Zhou, Feng, Loy. *Flare7K: A Phenomenological Nighttime Flare Removal Dataset.*
  NeurIPS 2022 Datasets and Benchmarks Track.
- Dai, Li, Zhou, Feng, Luo, Loy. *Flare7K++: Mixing Synthetic and Real Datasets for
  Nighttime Flare Removal and Beyond.* arXiv:2306.04236, 2023. Vihaan's handover lists a
  TPAMI 2024 version; confirm the venue before citing.

The README also lists a CVPR 2023 paper on reflective flare (optical-centre symmetry
prior). It is not the model used here, so cite it only if the paper discusses it.

## Checkpoint provenance

- **No published checksum exists.** The Flare7K README links the Flare7K++ checkpoint
  only through Google Drive and Baidu Netdisk (README lines 36 and 95). No SHA-256, MD5 or
  other hash appears anywhere in the repository (searched all `.md`, `.txt`, `.py`, `.yml`).
- **The local file is a repacked copy and cannot be byte-verified.** It arrived as
  `archive/`, the unpacked contents of a `.pth` file (`data.pkl`, `version`, 308 tensor
  files). It was repacked into `models/restoration/net_g_last.pth` (82,580,582 bytes,
  SHA-256 `8FDF489AAF89D58284F98C838449F73200F0B3B7FD32C28733C0B6A66FE14F1C`).
  Vihaan's README records 82,605,273 bytes and SHA-256
  `75F0FC77AB43703C7A9C7876621F8A651D6CE3A0CFB7C6E2377B3C8E2331B0E2` for his copy. The
  difference comes from zip padding in the repack, not the weights.
- **What was verified instead:** the file loads with `weights_only=True` and gives 308
  tensors and 20,547,616 parameters. Vihaan's adapter loads it into the Uformer with
  `strict=True`, and his demo reproduces his reported result to within 0.034 dB PSNR and
  0.0008 SSIM on the same input.
- **To byte-verify:** get the original `.pth` from Vihaan, or download from the Google
  Drive link above. Then compare its SHA-256 with his README.
