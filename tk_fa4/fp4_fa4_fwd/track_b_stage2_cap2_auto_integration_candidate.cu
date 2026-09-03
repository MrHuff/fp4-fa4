// Candidate-only build boundary for the cap2 automatic-dispatch integration.
// This file is compiled directly; it never includes or modifies tk_fa4.cu.
#define TK_FA4_FORWARD_ONLY_BUILD 1
#define TK_FA4_TRACK_B_STAGE2_CAP2_AUTO_INTEGRATION_BUILD 1

#include "fp4_fa4_fwd_experiments.cu"
