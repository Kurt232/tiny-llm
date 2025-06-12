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
        N, L, H, D = x.shape

        if offset is None:
            posi = mx.arange(L)
        else:
            if isinstance(offset, slice):
                offset = [offset]
            posi = []
            for offset in offset:
                start = offset.start or 0
                step = offset.step or 1
                stop = offset.stop or start + L
                posi.append(mx.arange(start, stop, step))
            posi = mx.concatenate(posi, axis=0)[:L]

        cos = self.cos_freqs[posi].reshape(1, L, 1, self.half_dims)
        sin = self.sin_freqs[posi].reshape(1, L, 1, self.half_dims)

        if self.traditional:
            x = x.reshape(N, L, H, self.half_dims, 2)  # N, H, L, D//2, 2
            x0 = x[..., 0]
            x1 = x[..., 1]
        else:
            x0 = x[..., : self.half_dims]
            x1 = x[..., self.half_dims :]

        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos

        if self.traditional:
            o = mx.stack([o0, o1], axis=-1).reshape(N, L, H, D)
        else:
            o = mx.concatenate([o0, o1], axis=-1)
        return o
