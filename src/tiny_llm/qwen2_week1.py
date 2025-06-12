import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0
        self.scale = mx.rsqrt(self.head_dim)
        assert wq.shape == (hidden_size, num_heads * self.head_dim)
        assert wk.shape == (num_kv_heads * self.head_dim, hidden_size)
        assert wv.shape == (num_kv_heads * self.head_dim, hidden_size)
        assert wo.shape == (num_heads * self.head_dim, hidden_size)

        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv

        self.rope = RoPE(
            dims=self.head_dim, seq_len=max_seq_len, base=theta, traditional=False
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        """
        ```
        x: B, L, E
        q = linear(x, wq, bq) -> B, L, H_q, D
        k = linear(x, wk, bk) -> B, L, H, D
        v = linear(x, wv, bv) -> B, L, H, D
        q = rope(q, offset=slice(offset, offset + L))
        k = rope(k, offset=slice(offset, offset + L))
        (transpose as needed)
        x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, L, H_q, D ; Do this at float32 precision
        (transpose as needed)
        x = linear(x, wo) -> B, L, E
        ```
        """
        B, L, _ = x.shape
        q = linear(x, self.wq, self.bq).reshape(B, L, self.num_heads, self.head_dim)
        k = linear(x, self.wk, self.bk).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = linear(x, self.wv, self.bv).reshape(B, L, self.num_kv_heads, self.head_dim)
        q = self.rope(q, offset=slice(offset, offset + L))
        k = self.rope(k, offset=slice(offset, offset + L))
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        x = scaled_dot_product_attention_grouped(
            q,
            k,
            v,
            scale=self.scale,
            mask=mask,
        )
        x = x.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        return linear(x, self.wo)


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        return linear(silu(linear(x, self.w_gate)) * linear(x, self.w_up), self.w_down)


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.mha = Qwen2MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            bq=bq,
            bk=bk,
            bv=bv,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.mlp = Qwen2MLP(
            dim=hidden_size,
            hidden_dim=intermediate_size,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
        )
        self.input_layernorm = RMSNorm(
            dim=hidden_size, weight=w_input_layernorm, eps=rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            dim=hidden_size, weight=w_post_attention_layernorm, eps=rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        x = x + self.mha(self.input_layernorm(x), offset, mask=mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        self.args = mlx_model.args
        precision = mx.float16
        self.precision = precision

        self.embed_tokens = Embedding(
            self.args.vocab_size,
            self.args.hidden_size,
            dequantize_linear(mlx_model.model.embed_tokens),
        )
        self.layers = [
            Qwen2TransformerBlock(
                num_attention_heads=self.args.num_attention_heads,
                num_kv_heads=self.args.num_key_value_heads,
                hidden_size=self.args.hidden_size,
                intermediate_size=self.args.intermediate_size,
                rms_norm_eps=self.args.rms_norm_eps,
                wq=dequantize_linear(layer.self_attn.q_proj),
                wk=dequantize_linear(layer.self_attn.k_proj),
                wv=dequantize_linear(layer.self_attn.v_proj),
                wo=dequantize_linear(layer.self_attn.o_proj),
                bq=layer.self_attn.q_proj.bias,
                bk=layer.self_attn.k_proj.bias,
                bv=layer.self_attn.v_proj.bias,
                w_gate=dequantize_linear(layer.mlp.gate_proj),
                w_up=dequantize_linear(layer.mlp.up_proj),
                w_down=dequantize_linear(layer.mlp.down_proj),
                w_input_layernorm=layer.input_layernorm.weight,
                w_post_attention_layernorm=layer.post_attention_layernorm.weight,
                max_seq_len=self.args.max_position_embeddings,
                theta=self.args.rope_theta,
            )
            for layer in mlx_model.model.layers
        ]
        self.norm = RMSNorm(
            self.args.hidden_size, mlx_model.model.norm.weight, self.args.rms_norm_eps
        )
        if not self.args.tie_word_embeddings:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head)

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        for layer in self.layers:
            h = layer(h, offset, mask="causal" if h.shape[1] > 1 else None)
        h = self.norm(h)
        if self.args.tie_word_embeddings:
            return self.embed_tokens.as_linear(h)
        else:
            return linear(h, self.w_lm_head, bias=False)
