# Copied logic from https://github.com/mit-han-lab/llm-awq/blob/f084f40bd996f3cf3a0633c1ad7d9d476c318aaa/awq/quantize/qmodule.py
import torch
import torch.nn as nn
from text_generation_server.utils.packing_utils import dequantize_gemm,get_best_device

try:
    import awq_ext  # with CUDA kernels
    AWQ_INSTALLED = True
except:
    AWQ_INSTALLED = False

# class ScaledActivation(nn.Module):
#     def __init__(self, module, scales):
#         super().__init__()
#         self.act = module
#         self.scales = nn.Parameter(scales.data)
#     
#     def forward(self, x):
#         return self.act(x) / self.scales.view(1, 1, -1).to(x.device)


class WQLinear(nn.Module):
    def __init__(self, w_bit, group_size, qweight, qzeros, scales, bias):
        super().__init__()
        
        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = qweight.shape[0]
        self.out_features = qweight.shape[1] * 32 // w_bit

        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else self.in_features
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert self.out_features % (32 // self.w_bit) == 0

        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        if bias:
            self.bias = bias
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None
    ):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return awq_linear

        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scale_zeros = zeros * scales

        awq_linear.scales = scales.clone().half()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        pack_num = 32 // awq_linear.w_bit

        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(
                torch.round(
                    (linear.weight.data[:, idx] + scale_zeros[idx // group_size])
                    / awq_linear.scales[idx // group_size]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)

        best_device = get_best_device()

        # Avoid: The operator 'aten::__lshift__.Scalar' is not currently implemented for the MPS device
        if "mps" in best_device:
            intweight = intweight.to("cpu")

        qweight = torch.zeros(
            (intweight.shape[0], intweight.shape[1] // 32 * awq_linear.w_bit),
            dtype=torch.int32,
            device=intweight.device,
        )

        for col in range(intweight.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
        awq_linear.qweight = qweight

        zeros = zeros.to(dtype=torch.int32, device=best_device)

        if "mps" in best_device:
            zeros = zeros.to("cpu")
        
        qzeros = torch.zeros(
            (zeros.shape[0], zeros.shape[1] // 32 * awq_linear.w_bit),
            dtype=torch.int32,
            device=zeros.device,
        )

        for col in range(zeros.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros

        return awq_linear
    
    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        input_dtype = x.dtype
        if input_dtype != torch.float16:
            x = x.half()

        if AWQ_INSTALLED:
            FP16_MATMUL_HEURISTIC_CONDITION = x.shape[0]*x.shape[1] >= 1024

            if FP16_MATMUL_HEURISTIC_CONDITION:
                out = awq_ext.dequantize_weights_cuda(
                    self.qweight,
                    self.scales,
                    self.qzeros,
                    0,
                    0,
                    0,
                    False
                )
                out = torch.matmul(x, out)
            else:
                out = awq_ext.gemm_forward_cuda(
                    x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, 8
                )
        else:
            out = dequantize_gemm(
                self.qweight,
                self.qzeros,
                self.scales,
                self.w_bit,
                self.group_size
            )
            out = torch.matmul(x, out)

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)   