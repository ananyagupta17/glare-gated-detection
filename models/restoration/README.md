# Restoration checkpoint

`net_g_last.pth` is the pretrained Flare7K++ Uformer generator checkpoint used
by `src/restore.py`.

- Size: 82,605,273 bytes
- SHA-256: `75F0FC77AB43703C7A9C7876621F8A651D6CE3A0CFB7C6E2377B3C8E2331B0E2`
- Distribution: shared separately; intentionally ignored by Git
- Upstream implementation: <https://github.com/ykdai/Flare7K>
- Associated paper: Dai et al., *Flare7K++: Mixing Synthetic and Real Datasets
  for Nighttime Flare Removal and Beyond*, IEEE TPAMI, 2024.
- Licence: see `THIRD_PARTY_LICENSES/Flare7K-LICENSE.txt` at the repository root.

After cloning this repository, copy the separately shared file to
`models/restoration/net_g_last.pth` and verify its checksum. The external
Flare7K source checkout is also required; see the root README and
`HANDOVER_RESTORATION.md`.
