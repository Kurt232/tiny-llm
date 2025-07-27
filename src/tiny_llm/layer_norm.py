import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight.astype(mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        assert x.shape[-1] == self.dim
        dtype = x.dtype
        x = x.astype(mx.float32)
        x = (
            x
            * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
            * self.weight
        )
        return x.astype(dtype)
