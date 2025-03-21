// Copyright 2016 rust-punkt developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use punkt::params::Standard;
use punkt::WordTokenizer;
use punkt::{SentenceTokenizer, TrainingData};

fn word_tokenizer_bench(c: &mut Criterion) {
  let mut group = c.benchmark_group("WordTokenizer");

  // Short document
  let short_doc = include_str!("../test/raw/sigma-wiki.txt");
  group.bench_function("short_doc", |b| {
    b.iter(|| {
      let t: WordTokenizer<Standard> = WordTokenizer::new(black_box(short_doc));
      let _: Vec<_> = t.collect();
    })
  });

  // Medium document
  let medium_doc = include_str!("../test/raw/npr-article-01.txt");
  group.bench_function("medium_doc", |b| {
    b.iter(|| {
      let t: WordTokenizer<Standard> = WordTokenizer::new(black_box(medium_doc));
      let _: Vec<_> = t.collect();
    })
  });

  // Long document
  let long_doc = include_str!("../test/raw/the-sayings-of-confucius.txt");
  group.bench_function("long_doc", |b| {
    b.iter(|| {
      let t: WordTokenizer<Standard> = WordTokenizer::new(black_box(long_doc));
      let _: Vec<_> = t.collect();
    })
  });

  // Very long document
  let very_long_doc = include_str!("../test/raw/pride-and-prejudice.txt");
  group.bench_function("very_long_doc", |b| {
    b.iter(|| {
      let t: WordTokenizer<Standard> = WordTokenizer::new(black_box(very_long_doc));
      let _: Vec<_> = t.collect();
    })
  });

  group.finish();
}

fn sentence_tokenizer_bench(c: &mut Criterion) {
  let mut group = c.benchmark_group("SentenceTokenizer");

  // Short document
  let short_doc = include_str!("../test/raw/sigma-wiki.txt");
  let short_data = TrainingData::english();
  group.bench_function("short_doc", |b| {
    b.iter(|| {
      let t = SentenceTokenizer::<Standard>::new(black_box(short_doc), &short_data);
      let _: Vec<_> = t.collect();
    })
  });

  // Medium document
  let medium_doc = include_str!("../test/raw/npr-article-01.txt");
  let medium_data = TrainingData::english();
  group.bench_function("medium_doc", |b| {
    b.iter(|| {
      let t = SentenceTokenizer::<Standard>::new(black_box(medium_doc), &medium_data);
      let _: Vec<_> = t.collect();
    })
  });

  // Long document
  let long_doc = include_str!("../test/raw/pride-and-prejudice.txt");
  let long_data = TrainingData::english();
  group.bench_function("long_doc", |b| {
    b.iter(|| {
      let t = SentenceTokenizer::<Standard>::new(black_box(long_doc), &long_data);
      let _: Vec<_> = t.collect();
    })
  });

  group.finish();
}

criterion_group!(benches, word_tokenizer_bench, sentence_tokenizer_bench);
criterion_main!(benches);
