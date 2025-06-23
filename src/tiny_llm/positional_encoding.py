import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.traditional = traditional
        assert dims % 2 == 0
        self.half_dims = dims // 2

        exp = mx.array([[(-2 * i / dims) for i in range(self.half_dims)]])  # 1, D//2
        theta = base**exp
        self.cos_freqs = mx.cos(
            mx.matmul(mx.arange(seq_len).reshape(-1, 1), theta)
        )  # L, D//2
        self.sin_freqs = mx.sin(
            mx.matmul(mx.arange(seq_len).reshape(-1, 1), theta)
        )  # L, D//2

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        """

        x: N, L, H, D
        """
        N, S, H, D = x.shape
        if offset is not None:
            if isinstance(offset, slice):
                assert offset.stop - offset.start == S, f"offset must be of length {S}"
            elif isinstance(offset, list):
                assert len(offset) == N, (
                    f"offsets must have the same length as batch size {N}"
                )
                for o in offset:
                    assert o.stop - o.start == S, f"offset must be of length {S}"
                offset = mx.array([list(range(i.start, i.stop)) for i in offset])
        cos = self.cos_freqs[:S, :] if offset is None else self.cos_freqs[offset, :]
        sin = self.sin_freqs[:S, :] if offset is None else self.sin_freqs[offset, :]
        # reshape x: (b, s, n_heads, head_dim // 2, 2)
        if self.traditional:
            x = x.reshape(N, S, H, self.half_dims, 2)
            x0 = x[..., 0]
            x1 = x[..., 1]
        else:
            x0 = x[..., : self.half_dims]
            x1 = x[..., self.half_dims :]
        # reshape basis: (1, s, 1, dims // 2, 2)
        cos = cos.reshape(-1, S, 1, self.half_dims)
        sin = sin.reshape(-1, S, 1, self.half_dims)

        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        if self.traditional:
            o = mx.stack([o0, o1], axis=-1).reshape(N, S, H, D)
        else:
            o = mx.concatenate([o0, o1], axis=-1)
        return o
