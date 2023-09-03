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

These results have the same biases as the official Google T5  implementation.

## How do we fix it?

### Encouraging randomness in `length<=30`

One idea is:

```diff
- # avoid degeneracy by ensuring positive number of noise spans
- num_noise_spans: int = max(num_noise_spans, 1)
+ # avoid degeneracy by ensuring segmentable number of noise spans
+ num_noise_spans: int = max(num_noise_spans, 2)
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

I am not sure whether it preserves other guarantees like noise density and mean noise span length, so requires a bit more thought.

### Eliminating end-of-sequence bias

No proposed solution yet; need to read the algorithm a bit more closely to understand what's causing this.