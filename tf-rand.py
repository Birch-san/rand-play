import tensorflow as tf
from tensorflow import Tensor
import seqio

# from Google Research:
# https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682
def random_spans_noise_mask(
  length: int,
  noise_density: float,
  seeds: Tensor,
  mean_noise_span_length=3.0
) -> Tensor:
  """Noise mask consisting of random spans of noise tokens.

  The number of noise tokens and the number of noise spans and non-noise spans
  are determined deterministically as follows:

    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(
       num_noise_tokens / mean_noise_span_length)

  Spans alternate between non-noise and noise, beginning with non-noise.
  Subject to the above restrictions, all masks are equally likely.

  Args:
    length: an int32 scalar (length of the incoming token sequence)
    noise_density: a float - approximate density of output mask
    seeds: an int32 Tensor, shaped (2, 2)
    mean_noise_span_length: a number

  Returns:
    a boolean tensor with shape [length]
  """

  orig_length = length
  # increase length to avoid degeneracy
  length = tf.maximum(length, 2)
  def to_int(x):
    return tf.cast(x, tf.int32)
  def to_float(x):
    return tf.cast(x, tf.float32)
  num_noise_tokens = to_int(tf.round(to_float(length) * noise_density))
  # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
  num_noise_tokens = tf.minimum(tf.maximum(num_noise_tokens, 1), length - 1)
  num_noise_spans = to_int(
      tf.round(to_float(num_noise_tokens) / mean_noise_span_length))
  # avoid degeneracy by ensuring positive number of noise spans
  num_noise_spans = tf.maximum(num_noise_spans, 1)
  num_nonnoise_tokens = length - num_noise_tokens
  # pick the lengths of the noise spans and the non-noise spans
  def _random_segmentation(num_items, num_segments, seed: int):
    """Partition a sequence of items randomly into non-empty segments.

    Args:
      num_items: an integer scalar > 0
      num_segments: an integer scalar in [1, num_items]
      seed: an integer seed
    Returns:
      a Tensor with shape [num_segments] containing positive integers that add
      up to num_items
    """
    first_in_segment = tf.pad(
        seqio.stateless_shuffle(
            to_int(tf.range(num_items - 1) < num_segments - 1),
            seed),
        [[1, 0]])
    segment_id = tf.cumsum(first_in_segment)
    segment_length = tf.math.segment_sum(tf.ones_like(segment_id), segment_id)
    return segment_length
  noise_span_lengths = _random_segmentation(
      num_noise_tokens, num_noise_spans, seeds[0])
  nonnoise_span_lengths = _random_segmentation(
      num_nonnoise_tokens, num_noise_spans, seeds[1])
  interleaved_span_lengths = tf.reshape(
      tf.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
      [num_noise_spans * 2])
  span_starts = tf.cumsum(interleaved_span_lengths)[:-1]
  span_start_indicator = tf.math.unsorted_segment_sum(
      tf.ones_like(span_starts), span_starts, length)
  span_num = tf.cumsum(span_start_indicator)
  is_noise = tf.equal(span_num % 2, 1)
  return is_noise[:orig_length]

noise_density=.15
mean_noise_span_length=3.
attempts=10
for length in [30, 31]:
  print(f'generating {attempts} noise masks for seq length {length}:')
  for _ in range(attempts):
    seeds=tf.random.uniform(shape=(2, 2), minval=1, maxval=1024, dtype=tf.int32)
    mask: Tensor = random_spans_noise_mask(length=length, noise_density=noise_density, seeds=seeds, mean_noise_span_length=mean_noise_span_length)
    print(''.join([str(x) for x in mask.__array__(int)]))