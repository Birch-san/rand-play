# T5 random spans investigation

Is T5's `random_spans_noise_mask` biased (i.e. always masks the end of the sentence)?  
Is it random at all for short sequences (`length<=30`)?

Let's run it and find out.

## TensorFlow implementation from Google

From [`google-research/text-to-text-transfer-transformer`](https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682).

### Setup

_Note: requires tensorflow-text, which I believe is only distributed for Python 3.10, and for which there is no official ARM macOS distribution._

```bash
pip install -r requirements.tf.txt
```

### Usage

```bash
python -m tf-rand
```

```
generating 10 noise masks for seq length 30:
000000000000000000000000001111
000000000000000000000000001111
000000000000000000000000001111
000000000000000000000000001111
000000000000000000000000001111
000000000000000000000000001111
000000000000000000000000001111
000000000000000000000000001111
000000000000000000000000001111
000000000000000000000000001111
generating 10 noise masks for seq length 31:
0000000000000000000001000001111
0000000000000000001000000001111
0000000000000000001110000000011
0000000000000000000000001100111
0000000000000000000000011000111
0000000111000000000000000000011
0000000110000000000000000000111
0000000000000000000010000001111
0111100000000000000000000000001
0000001110000000000000000000011
```

0 = use original token  
1 = replace these tokens with a noise span

Notice how for `length<=30`: the random spans noise masks are not random.

We also observe a bias that always places a noise span at the end of the sequence.

## Numpy implementation from Huggingface

From [`huggingface/transformers`](https://github.com/huggingface/transformers/blob/0afa5071bd84e44301750fdc594e33db102cf374/examples/flax/language-modeling/run_t5_mlm_flax.py#L405).

### Setup

```bash
pip install -r requirements.np.txt
```

### Usage

```bash
python -m np-rand
```

```
generating 10 noise masks for seq length 30:
000000000000000000000000001111
000000000000000000000000001111
000000000000000000000000001111
000000000000000000000000001111
000000000000000000000000001111
000000000000000000000000001111
000000000000000000000000001111
000000000000000000000000001111
000000000000000000000000001111
000000000000000000000000001111
generating 10 noise masks for seq length 31:
0000000000000000000111100000001
0000000000000010000000000001111
0000000000000000000111100000001
0000111000000000000000000000011
0000000000000000000110000000111
0000000000000000000000011110001
0000000000000000110000000000111
0000000000000000000001111000001
0000000000000000000001000001111
0000000000000000111000000000011
```

These results have the same biases as the official Google T5 implementation.

_Note: HF's implementation encounters an out-of-bounds error at `length<4`, whereas Google's implementation does not._

## How do we fix it?

See [`np-rand-fix`](np-rand-fix.py) for my attempt at fixing `random_spans_noise_mask` (based off of HF's numpy implementation).

### Setup

```bash
pip install -r requirements.np.txt
```

### Usage

```bash
python -m np-rand-fix
```

```
generating 10 noise masks for seq length 30:
000001111000000000000000000000
000000000001111000000000000000
000000000000000001111000000000
111100000000000000000000000000
000000000000000000000000001111
000000011110000000000000000000
000001111000000000000000000000
000000000000000011110000000000
001111000000000000000000000000
000000000000000000000001111000
generating 10 noise masks for seq length 31:
0000000001111001000000000000000
0000000010000001111000000000000
0000000011100000000000000001100
0000011110000000000000000000100
0000000000111011000000000000000
0000000000000011110000000001000
0001110000000000011000000000000
0000110000111000000000000000000
0000110000111000000000000000000
0100000000000000000000011110000
```

### Encouraging randomness in `length<=30`

The root cause of the problem is that `_random_segmentation` only succeeds for `num_noise_spans>1`.

**[idea 1]: impose a minimum number of noise spans**

We could impose a minimum of 2 spans:

```diff
- # avoid degeneracy by ensuring positive number of noise spans
- num_noise_spans: int = max(num_noise_spans, 1)
+ # avoid degeneracy by ensuring segmentable number of noise spans
+ num_noise_spans: int = max(num_noise_spans, 2)
```

We'd also need to ensure we have at least as many noise tokens as we have spans:

```diff
- num_noise_tokens = int(np.round(length * noise_density))
+ num_noise_tokens = int(max(np.round(length * noise_density), 2))
```

This recovers randomness in the length=30 result:

```
generating 10 noise masks for seq length 30:
000000000000000000100000000111
000000000000000000000110000011
000000001110000000000000000001
000000000000010000000000000111
000000000011000000000000000011
000011100000000000000000000001
001000000000000000000000000111
011000000000000000000000000011
000011100000000000000000000001
000000000000110000000000000011
```

I think however it risks creating more noise than requested.

**[idea 2]: special case for `num_noise_spans==1`**

After computing the final value for `num_noise_spans`, we can add a special case for `num_noise_spans==1`:

```python
if num_noise_spans == 1:
  # we do not have a segmentable number of noise spans, so _random_segmentation would give a non-random result (puts span at end-of-sequence)
  mask: NDArray = np.zeros((length,), dtype=np.bool_)
  start_noise_ix: int = randint(0, length-1)
  noise_indices: NDArray = np.fmod(np.arange(start_noise_ix, start_noise_ix + num_noise_tokens), length)
  np.put_along_axis(mask, values=True, indices=noise_indices, axis=-1)
  return mask
```

This takes advantage of the fact that "one span" is a _far_ easier problem.  
We just write a run of `num_noise_tokens` starting at a random index.  
We use `np.fmod(…, length)` to enable wrap-around if we go past the end of the sequence.

This preserves the guarantee of writing the correct number of spans and noise tokens. It also fix's HF's out-of-bounds error for `length<4`.

### Eliminating end-of-sequence bias

We could roll the mask by a random offset along its row dim:

```diff
  mask: NDArray = is_noise[:orig_length]
+ mask = np.roll(mask, randint(0, mask.shape[-1]-1), axis=-1)
  return mask
```

This fixes the end-of-sequence bias:

```
generating 10 noise masks for seq length 30:
000000000000000000000000101110
000000000000000000001100001100
000000011000011000000000000000
111000000000000000010000000000
000010000000000000000000001110
000111000000000000000000100000
000000001000000000000111000000
000000000011000000110000000000
000000000000000000000001110100
000000000000011100000000000100
generating 10 noise masks for seq length 31:
0000000001100000000000011100000
0011100000000000110000000000000
1110110000000000000000000000000
0000000000000011110000100000000
1111000000000000000000000100000
0000101111000000000000000000000
0000000000001111000000001000000
0001111000000000000000000000010
0000000001100000000001110000000
0000000000001111000000000100000
```

Perhaps a smarter solution could be found by reading the algorithm a bit more closely to understand the root cause of the bias.