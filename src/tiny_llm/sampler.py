import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp < 0:
            raise ValueError(f"`temp` has to be a non-negative float, but is {temp}.")
        if top_p < 0 or top_p > 1:
            raise ValueError(
                f"`top_p` has to be a float in the [0, 1] interval, but is {top_p}."
            )

        if temp == 0:
            return mx.argmax(logprobs, axis=-1)

        if top_k is not None:
            vocab_size = logprobs.shape[-1]
            if not isinstance(top_k, int) or not (0 < top_k < vocab_size):
                raise ValueError(
                    f"`top_k` has to be an integer in the (0, {vocab_size}] interval,"
                    f" but is {top_k}."
                )
            mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
            logprobs = mx.put_along_axis(
                logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
            )

        # top-p
        # referenced implementation from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L449-L460
        probs = mx.exp(logprobs)
        # sort in ascending order
        sorted_indices = mx.argsort(logprobs, axis=-1)
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

        # Rearrange cumulative probs back to original order
        inverse_indices = mx.put_along_axis(
            mx.zeros_like(sorted_indices),
            sorted_indices,
            mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
            axis=-1,
        )
        cumulative_probs = mx.take_along_axis(
            cumulative_probs, inverse_indices, axis=-1
        )

        # select tokens with cumulative probs below threshold
        logprobs = mx.where(
            cumulative_probs > 1 - top_p,
            logprobs,
            -float("inf"),
        )

        return mx.random.categorical(logprobs * (1 / temp))

    return sample
