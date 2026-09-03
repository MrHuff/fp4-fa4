# Local GB200 D64 control

`b1_s32768_h24_d64_nvfp8.json` records the apples-to-apples local control for
the D64 specialization. The TK and HAO rows use NVFP4 QK with plain FP8 PV;
the BF16 row uses HAO's native BF16 forward path. All three rows use the same
seeded tensors and direct forward timing protocol.

The optimized TK path is compiled with shiftless FP8 mode 4. It uses 128
registers, one barrier, and has no local-memory spills.
