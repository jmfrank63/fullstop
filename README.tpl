# {{crate}}

[![Build Status](https://travis-ci.org/ferristseng/rust-fullstop.svg)](https://travis-ci.org/ferristseng/rust-fullstop)
[![Crates.io](https://img.shields.io/crates/v/fullstop.svg)](https://crates.io/crates/fullstop)
[![Docs.rs](https://docs.rs/fullstop/badge.svg)](https://docs.rs/fullstop/)

{{readme}}

## Benchmarks

Specs of my machine:

  * i5-4460 @ 3.20 x 4
  * 8 GB RAM
  * Fedora 20
  * SSD

```
test tokenizer::bench_sentence_tokenizer_train_on_document_long   ... bench: 129,877,668 ns/iter (+/- 6,935,294)
test tokenizer::bench_sentence_tokenizer_train_on_document_medium ... bench:     901,867 ns/iter (+/- 12,984)
test tokenizer::bench_sentence_tokenizer_train_on_document_short  ... bench:     702,976 ns/iter (+/- 13,554)
test tokenizer::word_tokenizer_bench_long                         ... bench:  14,897,528 ns/iter (+/- 689,138)
test tokenizer::word_tokenizer_bench_medium                       ... bench:     339,535 ns/iter (+/- 21,692)
test tokenizer::word_tokenizer_bench_short                        ... bench:     281,293 ns/iter (+/- 3,256)
test tokenizer::word_tokenizer_bench_very_long                    ... bench:  54,256,241 ns/iter (+/- 1,210,575)
test trainer::bench_trainer_long                                  ... bench:  27,674,731 ns/iter (+/- 550,338)
test trainer::bench_trainer_medium                                ... bench:     681,222 ns/iter (+/- 31,713)
test trainer::bench_trainer_short                                 ... bench:     527,203 ns/iter (+/- 11,354)
test trainer::bench_trainer_very_long                             ... bench:  98,221,585 ns/iter (+/- 5,297,733)

```

Python results for sentence tokenization, and training on the document (the first 3 tests mirrored from above):

The following script was used to benchmark NLTK.

  * `f0` is the contents of the file that is being tokenized.
  * `s` is an instance of a `PunktSentenceTokenizer`.
  * `timed` is the total time it takes to run `tests` number of tests.

*`False` is being passed into `tokenize` to prevent NLTK from aligning sentence boundaries. This functionality 
is currently unimplemented.*

```python
timed = timeit.timeit('s.train(f0); [s for s in s.tokenize(f0, False)]', 'from bench import s, f0', number=tests)
print(timed)
print(timed / tests)
```

```
long    - 1.3414202709775418 s   = 1.34142 x 10^9 ns ~ 10.3283365927x improvement 
medium  - 0.007250561956316233 s = 7.25056 x 10^6 ns ~ 8.03950245027x improvement
short   - 0.005532620595768094 s = 5.53262 x 10^6 ns ~ 7.870283759x   improvement
```

## License

Licensed under either of

 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
