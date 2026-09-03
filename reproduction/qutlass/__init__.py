import torch

def matmul_mxf4_bf16_tn(x, w, xs, ws, alpha):
    return torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16, device=x.device)

def matmul_nvf4_bf16_tn(x, w, xs, ws, alpha):
    return torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16, device=x.device)

def fusedQuantizeMx(x_flat, hadamard_matrix, return_mask=False):
    rows, cols = x_flat.shape[0], x_flat.shape[1] // 32
    padded_rows = ((rows + 128 - 1) // 128) * 128
    padded_cols = ((cols + 4 - 1) // 4) * 4
    xh_e2m1 = torch.empty(x_flat.shape[0], x_flat.shape[1] // 2, dtype=torch.uint8, device=x_flat.device)
    xh_e8m0 = torch.empty(padded_rows, padded_cols, dtype=torch.uint8, device=x_flat.device)
    clip_mask = torch.empty(*x_flat.shape[:-1], x_flat.size(-1) // 8, dtype=torch.uint8, device=x_flat.device) if return_mask else None
    if return_mask:
        return xh_e2m1, xh_e8m0, clip_mask
    else:
        return xh_e2m1, xh_e8m0

def backward_t_bf16(grad_output_flat, hadamard_matrix):
    xh_e2m1 = torch.empty(grad_output_flat.shape[1], grad_output_flat.shape[0] // 2, dtype=torch.uint8, device=grad_output_flat.device)
    xh_e8m0 = torch.empty(grad_output_flat.shape[1], grad_output_flat.shape[0] // 32, dtype=torch.uint8, device=grad_output_flat.device)
    return xh_e2m1, xh_e8m0

def backward_qt_bf16(x_e2m1, x_e8m0, h, alpha):
    xh_e2m1 = torch.empty(x_e2m1.shape[1] * 2, x_e2m1.shape[0] // 2, dtype=torch.uint8, device=h.device)
    xh_e8m0 = torch.empty(x_e8m0.shape[1] * 32, x_e8m0.shape[0] // 32, dtype=torch.uint8, device=h.device)
    return xh_e2m1, xh_e8m0

def matmul_mxf8_bf16_tn(x, w, xs, ws, alpha):
    return torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16, device=x.device)
