// Copyright 2016 rust-punkt developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Overview
//!
//! Implementation of Tibor Kiss' and Jan Strunk's Fullstop algorithm for sentence
//! tokenization. Results have been compared with small and large texts that have
//! been tokenized using NLTK.
//!
//! # Training
//!
//! Training data can be provided to a `SentenceTokenizer` for better
//! results. Data can be acquired manually by training with a `Trainer`,
//! or using already compiled data from NLTK (example: `TrainingData::english()`).
//!
//! # Typical Usage
//!
//! The fullstop algorithm allows you to derive all the necessary data to perform
//! sentence tokenization from the document itself.
//!
//! ```
//! # use fullstop::params::Standard;
//! # use fullstop::{Trainer, TrainingData, SentenceTokenizer};
//! #
//! # let doc = "I bought $5.50 worth of apples from the store. I gave them to my dog when I came home.";
//! let trainer: Trainer<Standard> = Trainer::new();
//! let mut data = TrainingData::new();
//!
//! trainer.train(doc, &mut data);
//!
//! for s in SentenceTokenizer::<Standard>::new(doc, &data) {
//!   println!("{:?}", s);
//! }
//! ```
//!
//! `rust-fullstop` also provides pretrained data that can be loaded for certain languages.
//!
//! ```
//! # #![allow(unused_variables)]
//! #
//! # use fullstop::TrainingData;
//! #
//! let data = TrainingData::english();
//! ```
//!
//! `rust-fullstop` also allows training data to be incrementally gathered.
//!
//! ```
//! # use fullstop::params::Standard;
//! # use fullstop::{Trainer, TrainingData, SentenceTokenizer};
//! #
//! # let docs = ["This is a sentence with a abbrev. in it."];
//! let trainer: Trainer<Standard> = Trainer::new();
//! let mut data = TrainingData::new();
//!
//! for d in docs.iter() {
//!   trainer.train(d, &mut data);
//!
//!   for s in SentenceTokenizer::<Standard>::new(d, &data) {
//!     println!("{:?}", s);
//!   }
//! }
//! ```
//!
//! # Customization
//!
//! `rust-fullstop` exposes a number of traits to customize how the trainer, sentence tokenizer,
//! and internal tokenizers work. The default settings, which are nearly identical, to the
//! ones available in the Python library are available in `fullstop::params::Standard`.
//!
//! To modify only how the trainer works:
//!
//! ```
//! # use fullstop::params::*;
//! #
//! struct MyParams;
//!
//! impl DefinesInternalPunctuation for MyParams {}
//! impl DefinesNonPrefixCharacters for MyParams {}
//! impl DefinesNonWordCharacters for MyParams {}
//! impl DefinesPunctuation for MyParams {}
//! impl DefinesSentenceEndings for MyParams {}
//!
//! impl TrainerParameters for MyParams {
//!   const ABBREV_LOWER_BOUND: f64 = 0.3;
//!   const ABBREV_UPPER_BOUND: f64 = 8f64;
//!   const IGNORE_ABBREV_PENALTY: bool = false;
//!   const COLLOCATION_LOWER_BOUND: f64 = 7.88;
//!   const SENTENCE_STARTER_LOWER_BOUND: f64 = 35f64;
//!   const INCLUDE_ALL_COLLOCATIONS: bool = false;
//!   const INCLUDE_ABBREV_COLLOCATIONS: bool = true;
//!   const COLLOCATION_FREQUENCY_LOWER_BOUND: f64 = 0.8f64;
//! }
//! ```
//!
//! To fully modify how everything works:
//!
//! ```
//! # use fullstop::params::*;
//! #
//! struct MyParams;
//!
//! impl DefinesSentenceEndings for MyParams {
//!   // const SENTENCE_ENDINGS: &'static Set<char> = &phf_set![...];
//! }
//!
//! impl DefinesInternalPunctuation for MyParams {
//!   // const INTERNAL_PUNCTUATION: &'static Set<char> = &phf_set![...];
//! }
//!
//! impl DefinesNonWordCharacters for MyParams {
//!   // const NONWORD_CHARS: &'static Set<char> = &phf_set![...];
//! }
//!
//! impl DefinesPunctuation for MyParams {
//!   // const PUNCTUATION: &'static Set<char> = &phf_set![...];
//! }
//!
//! impl DefinesNonPrefixCharacters for MyParams {
//!   // const NONPREFIX_CHARS: &'static Set<char> = &phf_set![...];
//! }
//!
//! impl TrainerParameters for MyParams {
//!   // const ABBREV_LOWER_BOUND: f64 = ...;
//!   // const ABBREV_UPPER_BOUND: f64 = ...;
//!   // const IGNORE_ABBREV_PENALTY: bool = ...;
//!   // const COLLOCATION_LOWER_BOUND: f64 = ...;
//!   // const SENTENCE_STARTER_LOWER_BOUND: f64 = ...;
//!   // const INCLUDE_ALL_COLLOCATIONS: bool = ...;
//!   // const INCLUDE_ABBREV_COLLOCATIONS: bool = true;
//!   // const COLLOCATION_FREQUENCY_LOWER_BOUND: f64 = ...;
//! }
//! ```

mod freq_dist;
mod prelude;
mod token;
mod tokenizer;
mod trainer;
mod util;

pub use tokenizer::WordTokenizer;
pub use tokenizer::{SentenceByteOffsetTokenizer, SentenceTokenizer};
pub use trainer::{Trainer, TrainingData};

/// Contains traits for configuring all tokenizers, and the trainer. Also
/// contains default parameters for tokenizers, and the trainer.
pub mod params {
    pub use crate::prelude::{
        DefinesInternalPunctuation, DefinesNonPrefixCharacters, DefinesNonWordCharacters,
        DefinesPunctuation, DefinesSentenceEndings, Set, Standard, TrainerParameters,
    };
}

#[cfg(test)]
fn get_test_scenarios(dir_path: &str, raw_path: &str) -> Vec<(Vec<String>, String, String)> {
    use std::fs;
    use std::io::Read;
    use std::path::Path;

    use walkdir::WalkDir;

    let mut tests = Vec::new();

    for path in WalkDir::new(dir_path) {
        let entry = path.unwrap();
        let fpath = entry.path();

        if fpath.is_file() {
            let mut exp_strb = String::new();
            let mut raw_strb = String::new();

            // Files in the directory with raw articles must match the file names of
            // articles in the directory with test outcomes.
            let rawp = Path::new(raw_path).join(fpath.file_name().unwrap());

            fs::File::open(fpath)
                .expect("Failed to open a file")
                .read_to_string(&mut exp_strb)
                .expect("Failed to read to string");
            fs::File::open(&rawp)
                .expect("Failed to open a file")
                .read_to_string(&mut raw_strb)
                .expect("Failed to read to string");

            // Expected results, split by newlines.
            let exps: Vec<String> = exp_strb.split('\n').map(|s| s.to_string()).collect();

            tests.push((exps, raw_strb, format!("{:?}", fpath.file_name().unwrap())));
        }
    }

    tests // Returns (Expected cases, File contents, File name)
}
