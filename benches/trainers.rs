// Copyright 2016 rust-punkt developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use fullstop::params::Standard;
use fullstop::{Trainer, TrainingData};

fn trainer_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("Trainer");

    // Short document
    let short_doc = include_str!("../test/raw/sigma-wiki.txt");
    group.bench_function("short_doc", |b| {
        b.iter(|| {
            let mut data = TrainingData::new();
            let trainer: Trainer<Standard> = Trainer::new();
            trainer.train(black_box(short_doc), &mut data);
        })
    });

    // Medium document
    let medium_doc = include_str!("../test/raw/npr-article-01.txt");
    group.bench_function("medium_doc", |b| {
        b.iter(|| {
            let mut data = TrainingData::new();
            let trainer: Trainer<Standard> = Trainer::new();
            trainer.train(black_box(medium_doc), &mut data);
        })
    });

    // Long document
    let long_doc = include_str!("../test/raw/the-sayings-of-confucius.txt");
    group.bench_function("long_doc", |b| {
        b.iter(|| {
            let mut data = TrainingData::new();
            let trainer: Trainer<Standard> = Trainer::new();
            trainer.train(black_box(long_doc), &mut data);
        })
    });

    // Very long document
    let very_long_doc = include_str!("../test/raw/pride-and-prejudice.txt");
    group.bench_function("very_long_doc", |b| {
        b.iter(|| {
            let mut data = TrainingData::new();
            let trainer: Trainer<Standard> = Trainer::new();
            trainer.train(black_box(very_long_doc), &mut data);
        })
    });

    group.finish();
}

criterion_group!(benches, trainer_bench);
criterion_main!(benches);
