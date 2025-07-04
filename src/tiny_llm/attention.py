import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    """Scaled Dot Product Attention

    Because we are always using the attention layer within the multi-head attention layer, the actual tensor shape when serving
    the model will be:

    ```
    key: B x H x L x D
    value: B x H x L x D
    query: B x H x L x D
    output: B x H x L x D
    mask: B x H x L x L
    ```
    """
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor  # B x H x L x L
    if mask is not None:
        scores += mask

    return mx.matmul(softmax(scores, axis=-1), value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0
        self.scale = mx.rsqrt(self.head_dim)
        assert wq.shape == (hidden_size, num_heads * self.head_dim)
        assert wk.shape == (num_heads * self.head_dim, hidden_size)
        assert wv.shape == (num_heads * self.head_dim, hidden_size)
        assert wo.shape == (num_heads * self.head_dim, hidden_size)

        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        _bs = query.shape[0]
        proj_q = (
            linear(query, self.wq)
            .reshape(_bs, -1, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        proj_k = (
            linear(key, self.wk)
            .reshape(_bs, -1, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        proj_v = (
            linear(value, self.wv)
            .reshape(_bs, -1, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        o = scaled_dot_product_attention_simple(
            proj_q, proj_k, proj_v, scale=self.scale, mask=mask
        )
        o = o.transpose(0, 2, 1, 3).reshape(_bs, -1, self.hidden_size)

        return linear(o, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    full = mx.ones((L, S), dtype=dtype) * -mx.inf
    mask = mx.triu(full, k=(S - L + 1))
    return mask


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    """
    query: B x H_q x L x D
    key: B x H x S x D
    value: B x H x S x D
    output: B x H_q x L x D
    mask: B x H x L x S or "causal"

    output: B x H_q x L x D
    """
    expected_shape = query.shape

    H_q, L, D = query.shape[-3:]
    H, S, _ = key.shape[-3:]

    n_repeats = H_q // H

    query = query.reshape(
        -1, H, n_repeats, L, D
    )  # leverage broadcasting instead of repeating the K and V tensors
    key = key.reshape(-1, H, 1, S, D)  # ! -1, 1, H, S, D is wrong
    value = value.reshape(-1, H, 1, S, D)

    factor = mx.rsqrt(D) if scale is None else scale
    scores = (
        mx.matmul(query, key.swapaxes(-2, -1)) * factor
    )  # Broadcasting should handle the head repetition implicitly.
    if mask is not None:
        if mask == "causal":
            mask = causal_mask(L, S, scores.dtype).reshape(1, 1, 1, L, S)
        else:
            mask = mask.reshape(-1, H, n_repeats, L, S)
        scores += mask

    o = mx.matmul(softmax(scores, axis=-1), value)

    return o.reshape(expected_shape)


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
