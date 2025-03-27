// Copyright 2016 rust-punkt developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::UnsafeCell;
use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::default::Default;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::Deref;
use std::str::FromStr;

use serde_json::Value;

use crate::freq_dist::FrequencyDistribution;
use crate::prelude::{
    DefinesNonPrefixCharacters, DefinesNonWordCharacters, OrthographicContext, OrthographyPosition,
    TrainerParameters,
};
use crate::token::Token;
use crate::tokenizer::WordTokenizer;

/// A collocation is any pair of words that has a high likelihood of appearing
/// together.
#[derive(Debug, Eq)]
pub struct Collocation<T>
where
    T: Deref<Target = Token>,
{
    l: T,
    r: T,
}

impl<T> Collocation<T>
where
    T: Deref<Target = Token>,
{
    /// Create a new collocation from two tokens
    #[inline]
    pub fn new(l: T, r: T) -> Collocation<T> {
        Collocation { l, r }
    }

    /// Returns the left token of the collocation
    #[inline]
    pub fn left(&self) -> &T {
        &self.l
    }

    /// Returns the right token of the collocation
    #[inline]
    pub fn right(&self) -> &T {
        &self.r
    }
}

impl<T> Hash for Collocation<T>
where
    T: Deref<Target = Token>,
{
    #[inline]
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        (*self.l).typ_without_period().hash(state);
        (*self.r).typ_without_break_or_period().hash(state);
    }
}

impl<T> PartialEq for Collocation<T>
where
    T: Deref<Target = Token>,
{
    #[inline]
    fn eq(&self, x: &Collocation<T>) -> bool {
        (*self.l).typ_without_period() == (*x.l).typ_without_period()
            && (*self.r).typ_without_break_or_period() == (*x.r).typ_without_break_or_period()
    }
}

/// Stores data that was obtained during training.
///
/// # Examples
///
/// Precompiled data can be loaded via a language specific constructor.
///
/// ```
/// # use fullstop::TrainingData;
/// #
/// let eng_data = TrainingData::english();
/// let ger_data = TrainingData::german();
///
/// assert!(eng_data.contains_abbrev("va"));
/// assert!(ger_data.contains_abbrev("crz"));
/// ```
#[derive(Debug, Default)]
pub struct TrainingData {
    abbrevs: HashSet<String>,
    collocations: HashMap<String, HashSet<String>>,
    sentence_starters: HashSet<String>,
    orthographic_context: HashMap<String, OrthographicContext>,
}

impl TrainingData {
    /// Creates a new, empty data object.
    #[inline]
    pub fn new() -> TrainingData {
        TrainingData {
            ..Default::default()
        }
    }

    /// Check if a token is considered to be an abbreviation.
    #[inline]
    pub fn contains_abbrev(&self, tok: &str) -> bool {
        self.abbrevs.contains(tok)
    }

    /// Insert a newly learned abbreviation.
    #[inline]
    fn insert_abbrev(&mut self, tok: &str) -> bool {
        if !self.contains_abbrev(tok) {
            self.abbrevs.insert(tok.to_lowercase())
        } else {
            false
        }
    }

    /// Removes a learned abbreviation.
    #[inline]
    fn remove_abbrev(&mut self, tok: &str) -> bool {
        self.abbrevs.remove(tok)
    }

    /// Check if a token is considered to be a token that commonly starts a
    /// sentence.
    #[inline]
    pub fn contains_sentence_starter(&self, tok: &str) -> bool {
        self.sentence_starters.contains(tok)
    }

    /// Insert a newly learned word that signifies the start of a sentence.
    #[inline]
    fn insert_sentence_starter(&mut self, tok: &str) -> bool {
        if !self.contains_sentence_starter(tok) {
            self.sentence_starters.insert(tok.to_string())
        } else {
            false
        }
    }

    /// Checks if a pair of words are commonly known to appear together.
    #[inline]
    pub fn contains_collocation(&self, left: &str, right: &str) -> bool {
        self.collocations
            .get(left)
            .map(|s| s.contains(right))
            .unwrap_or(false)
    }

    /// Insert a newly learned pair of words that frequently appear together.
    fn insert_collocation(&mut self, left: &str, right: &str) -> bool {
        if !self.collocations.contains_key(left) {
            self.collocations.insert(left.to_string(), HashSet::new());
        }

        if !self.collocations.get(left).unwrap().contains(right) {
            self.collocations
                .get_mut(left)
                .unwrap()
                .insert(right.to_string());
            true
        } else {
            false
        }
    }

    /// Insert or update the known orthographic context that a word commonly
    /// appears in.
    #[inline]
    fn insert_orthographic_context(&mut self, tok: &str, ctxt: OrthographicContext) -> bool {
        if let Some(c) = self.orthographic_context.get_mut(tok) {
            *c |= ctxt;
            return false;
        }

        self.orthographic_context.insert(tok.to_string(), ctxt);
        true
    }

    /// Gets the orthographic context for a token. Returns 0 if the token
    /// was not yet encountered.
    #[inline]
    pub fn get_orthographic_context(&self, tok: &str) -> u8 {
        *self.orthographic_context.get(tok).unwrap_or(&0)
    }
}

impl FromStr for TrainingData {
    type Err = &'static str;

    /// Deserializes JSON and loads the data into a new TrainingData object.
    fn from_str(s: &str) -> Result<TrainingData, &'static str> {
        let json: Value = serde_json::from_str(s).map_err(|_| "failed to parse JSON")?;
        let mut data = TrainingData::default();

        // Parse abbreviations
        if let Some(abbrevs) = json.get("abbrev_types").and_then(|v| v.as_array()) {
            for abbrev in abbrevs {
                if let Some(abbrev_str) = abbrev.as_str() {
                    data.insert_abbrev(abbrev_str);
                }
            }
        } else {
            return Err("failed to parse expected abbrev_types");
        }

        // Parse sentence starters
        if let Some(starters) = json.get("sentence_starters").and_then(|v| v.as_array()) {
            for starter in starters {
                if let Some(starter_str) = starter.as_str() {
                    data.insert_sentence_starter(starter_str);
                }
            }
        } else {
            return Err("failed to parse expected sentence_starters");
        }

        // Parse collocations
        if let Some(collocations) = json.get("collocations").and_then(|v| v.as_array()) {
            for collocation in collocations {
                if let Some(coll_array) = collocation.as_array() {
                    if coll_array.len() >= 2 {
                        if let (Some(left), Some(right)) =
                            (coll_array[0].as_str(), coll_array[1].as_str())
                        {
                            data.collocations
                                .entry(left.to_string())
                                .or_insert_with(HashSet::new)
                                .insert(right.to_string());
                        }
                    }
                }
            }
        } else {
            return Err("failed to parse collocations section");
        }

        // Parse orthographic context
        if let Some(contexts) = json.get("ortho_context").and_then(|v| v.as_object()) {
            for (key, value) in contexts {
                if let Some(context) = value.as_u64() {
                    data.orthographic_context.insert(key.clone(), context as u8);
                }
            }
        } else {
            return Err("failed to parse orthographic context section");
        }

        Ok(data)
    }
}

/// A trainer will build data about abbreviations, sentence starters,
/// collocations, and context that tokens appear in. The data is
/// used by the sentence tokenizer to determine if a period is likely
/// part of an abbreviation, or actually marks the termination of a sentence.
pub struct Trainer<P> {
    params: PhantomData<P>,
}

impl<P> Trainer<P>
where
    P: TrainerParameters + DefinesNonPrefixCharacters + DefinesNonWordCharacters,
{
    /// Creates a new Trainer.
    #[inline]
    pub fn new() -> Trainer<P> {
        Trainer {
            params: PhantomData,
        }
    }

    /// Train on a document. Does tokenization using a WordTokenizer.
    pub fn train(&self, doc: &str, data: &mut TrainingData) {
        let mut period_token_count: usize = 0;
        let mut sentence_break_count: usize = 0;

        // Collect tokens and wrap them in UnsafeCell for interior mutability
        let tokens_raw: Vec<Token> = WordTokenizer::<P>::new(doc).collect();
        // Pre-allocate UnsafeCell vector with capacity
        let mut tokens: Vec<UnsafeCell<Token>> = Vec::with_capacity(tokens_raw.len());
        for token in tokens_raw {
            // wrap in UnsafeCell for mutability
            tokens.push(UnsafeCell::new(token));
        }
        let mut type_fdist: FrequencyDistribution<&str> = FrequencyDistribution::new();
        let mut collocation_fdist: FrequencyDistribution<Collocation<&Token>> =
            FrequencyDistribution::new();
        let mut sentence_starter_fdist: FrequencyDistribution<&Token> =
            FrequencyDistribution::new();

        // Count tokens and build frequency distribution
        for token_cell in &tokens {
            let token = unsafe { &*token_cell.get() };
            if token.has_final_period() {
                period_token_count += 1;
            }
            type_fdist.insert(token.typ());
        }

        // Reclassify tokens as abbreviations if needed
        {
            // Create references to tokens for the iterator
            let token_refs: Vec<&Token> =
                tokens.iter().map(|cell| unsafe { &*cell.get() }).collect();

            let reclassify_iter: ReclassifyIterator<_, P> = ReclassifyIterator {
                iter: token_refs.iter().copied(),
                data,
                period_token_count,
                type_fdist: &mut type_fdist,
                params: PhantomData,
            };

            for (t, score) in reclassify_iter {
                if score >= P::ABBREV_LOWER_BOUND {
                    if t.has_final_period() {
                        // Need to mutate data while it's borrowed by the iterator
                        // This is safe because we're only modifying a HashSet within data
                        // which doesn't affect the iteration
                        unsafe {
                            let data_mut = &mut *(data as *const TrainingData as *mut TrainingData);
                            data_mut.insert_abbrev(t.typ_without_period());
                        }
                    }
                } else if !t.has_final_period() {
                    unsafe {
                        let data_mut = &mut *(data as *const TrainingData as *mut TrainingData);
                        data_mut.remove_abbrev(t.typ_without_period());
                    }
                }
            }
        }

        // First pass annotation - using UnsafeCell to mutate tokens safely
        for token_cell in &tokens {
            unsafe {
                let token_mut = &mut *token_cell.get();
                crate::util::annotate_first_pass::<P>(token_mut, data);
            }
        }

        // Update orthographic context
        {
            // Create references to tokens for the iterator
            let token_refs: Vec<&Token> =
                tokens.iter().map(|cell| unsafe { &*cell.get() }).collect();

            let token_with_context_iter = TokenWithContextIterator {
                iter: token_refs.iter().copied(),
                ctxt: OrthographyPosition::Internal,
            };

            for (t, ctxt) in token_with_context_iter {
                if ctxt != 0 {
                    data.insert_orthographic_context(t.typ_without_break_or_period(), ctxt);
                }
            }
        }

        // Count sentence breaks
        for token_cell in &tokens {
            let token = unsafe { &*token_cell.get() };
            if token.is_sentence_break() {
                sentence_break_count += 1;
            }
        }

        // Find potential collocations and sentence starters
        {
            // Create references to tokens for the iterator
            let token_refs: Vec<&Token> =
                tokens.iter().map(|cell| unsafe { &*cell.get() }).collect();

            let consecutive_token_iter = ConsecutiveItemIterator {
                iter: token_refs.iter().copied(),
                last: None,
            };

            for (lt, rt) in consecutive_token_iter {
                if let Some(cur) = rt {
                    if lt.has_final_period() {
                        if is_rare_abbrev_type::<P>(data, &type_fdist, lt, cur) {
                            data.insert_abbrev(lt.typ_without_period());
                        }

                        if is_potential_sentence_starter(cur, lt) {
                            sentence_starter_fdist.insert(cur);
                        }

                        if is_potential_collocation::<P>(lt, cur) {
                            collocation_fdist.insert(Collocation::new(lt, cur));
                        }
                    }
                }
            }
        }

        // Process sentence starters
        {
            let ss_iter: PotentialSentenceStartersIterator<_, P> =
                PotentialSentenceStartersIterator {
                    iter: sentence_starter_fdist.keys(),
                    sentence_break_count,
                    type_fdist: &type_fdist,
                    sentence_starter_fdist: &sentence_starter_fdist,
                    params: PhantomData,
                };

            for (tok, _) in ss_iter {
                data.insert_sentence_starter(tok.typ());
            }
        }

        // Process collocations
        {
            let clc_iter: PotentialCollocationsIterator<_, P> = PotentialCollocationsIterator {
                iter: collocation_fdist.keys(),
                data,
                type_fdist: &type_fdist,
                collocation_fdist: &collocation_fdist,
                params: PhantomData,
            };

            for (col, _) in clc_iter {
                // Need to mutate data while it's borrowed by the iterator
                // This is safe because we're only modifying a HashMap within data
                // which doesn't affect the iteration
                unsafe {
                    let data_mut = &mut *(data as *const TrainingData as *mut TrainingData);
                    data_mut.insert_collocation(
                        col.left().typ_without_period(),
                        col.right().typ_without_break_or_period(),
                    );
                }
            }
        }
    }
}

impl<P> Default for Trainer<P>
where
    P: TrainerParameters + DefinesNonPrefixCharacters + DefinesNonWordCharacters,
{
    fn default() -> Self {
        Self::new()
    }
}

fn is_rare_abbrev_type<P>(
    data: &TrainingData,
    type_fdist: &FrequencyDistribution<&str>,
    tok0: &Token,
    tok1: &Token,
) -> bool
where
    P: TrainerParameters,
{
    use crate::prelude::{BEG_UC, MID_UC};

    if tok0.is_abbrev() || !tok0.is_sentence_break() {
        false
    } else {
        let key = tok0.typ_without_break_or_period();
        let count = (type_fdist.get(key) + type_fdist.get(&key[..key.len() - 1])) as f64;

        // Already an abbreviation...
        if data.contains_abbrev(tok0.typ()) || count >= P::ABBREV_UPPER_BOUND {
            false
        } else if let Some(c) = tok1.typ().chars().next() {
            if P::is_internal_punctuation(c) {
                true
            } else if tok1.is_lowercase() {
                let ctxt = data.get_orthographic_context(tok1.typ_without_break_or_period());

        (ctxt & BEG_UC > 0) && (ctxt & MID_UC == 0)
            } else {
                false
            }
        } else {
            false
        }
    }
}

#[inline]
fn is_potential_sentence_starter(cur: &Token, prev: &Token) -> bool {
    prev.is_sentence_break() && !(prev.is_numeric() || prev.is_initial()) && cur.is_alphabetic()
}

#[inline]
fn is_potential_collocation<P>(tok0: &Token, tok1: &Token) -> bool
where
    P: TrainerParameters,
{
    P::INCLUDE_ALL_COLLOCATIONS
        || (P::INCLUDE_ABBREV_COLLOCATIONS && tok0.is_abbrev())
        || (tok0.is_sentence_break() && (tok0.is_numeric() || tok0.is_initial()))
            && tok0.is_non_punct()
            && tok1.is_non_punct()
}

/// Iterates over every token from the supplied iterator. Only returns
/// the ones that are 'not obviously' abbreviations. Also returns the associated
/// score of that token.
struct ReclassifyIterator<'b, I, P> {
    iter: I,
    data: &'b TrainingData,
    period_token_count: usize,
    type_fdist: &'b FrequencyDistribution<&'b str>,
    params: PhantomData<P>,
}

impl<'b, I, P> Iterator for ReclassifyIterator<'b, I, P>
where
    I: Iterator<Item = &'b Token>,
    P: TrainerParameters,
{
    type Item = (&'b Token, f64);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        for t in self.iter.by_ref() {
            if !t.is_non_punct() || t.is_numeric() {
                continue;
            }

            if t.has_final_period() {
                if self.data.contains_abbrev(t.typ()) {
                    continue;
                }
            } else if !self.data.contains_abbrev(t.typ()) {
                continue;
            }

            let num_periods = t
                .typ_without_period()
                .chars()
                .fold(0, |acc, c| if c == '.' { acc + 1 } else { acc })
                + 1;
            let num_nonperiods = t.typ_without_period().chars().count() - num_periods + 1;

            let count_with_period = self.type_fdist.get(t.typ_with_period());
            let count_without_period = self.type_fdist.get(t.typ_without_period());

            let likelihood = crate::util::dunning_log_likelihood(
                (count_with_period + count_without_period) as f64,
                self.period_token_count as f64,
                count_with_period as f64,
                self.type_fdist.sum_counts() as f64,
            );

            let f_length = (-(num_nonperiods as f64)).exp();
            let f_penalty = if P::IGNORE_ABBREV_PENALTY {
                0f64
            } else {
                (num_nonperiods as f64).powi(-(count_without_period as i32))
            };

            let score = likelihood * f_length * f_penalty * (num_periods as f64);

            return Some((t, score));
        }

        None
    }
}

struct TokenWithContextIterator<I> {
    iter: I,
    ctxt: OrthographyPosition,
}

impl<'a, I> Iterator for TokenWithContextIterator<I>
where
    I: Iterator<Item = &'a Token>,
{
    type Item = (&'a Token, OrthographicContext);

    #[inline]
    fn next(&mut self) -> Option<(&'a Token, OrthographicContext)> {
        match self.iter.next() {
            Some(t) => {
                if t.is_paragraph_start() && self.ctxt != OrthographyPosition::Unknown {
                    self.ctxt = OrthographyPosition::Initial;
                }

                if t.is_newline_start() && self.ctxt == OrthographyPosition::Internal {
                    self.ctxt = OrthographyPosition::Unknown;
                }

                let flag = *crate::prelude::ORTHO_MAP
                    .get(&(self.ctxt.as_byte() | t.first_case().as_byte()))
                    .unwrap_or(&0);

                if t.is_sentence_break() {
                    if !(t.is_numeric() || t.is_initial()) {
                        self.ctxt = OrthographyPosition::Initial;
                    } else {
                        self.ctxt = OrthographyPosition::Unknown;
                    }
                } else if t.is_ellipsis() || t.is_abbrev() {
                    self.ctxt = OrthographyPosition::Unknown;
                } else {
                    self.ctxt = OrthographyPosition::Internal;
                }

                Some((t, flag))
            }
            None => None,
        }
    }
}

struct PotentialCollocationsIterator<'b, I, P> {
    iter: I,
    data: &'b TrainingData,
    type_fdist: &'b FrequencyDistribution<&'b str>,
    collocation_fdist: &'b FrequencyDistribution<Collocation<&'b Token>>,
    params: PhantomData<P>,
}

impl<'a, I, P> Iterator for PotentialCollocationsIterator<'_, I, P>
where
    I: Iterator<Item = &'a Collocation<&'a Token>>,
    P: TrainerParameters,
{
    type Item = (&'a Collocation<&'a Token>, f64);

    #[inline]
    fn next(&mut self) -> Option<(&'a Collocation<&'a Token>, f64)> {
        for col in self.iter.by_ref() {
            if self
                .data
                .contains_sentence_starter(col.right().typ_without_break_or_period())
            {
                continue;
            }

            let count = self.collocation_fdist.get(col);

            let left_count = self.type_fdist.get(col.left().typ_without_period())
                + self.type_fdist.get(col.left().typ_with_period());
            let right_count = self.type_fdist.get(col.right().typ_without_period())
                + self.type_fdist.get(col.right().typ_with_period());

            if left_count > 1
                && right_count > 1
                && P::COLLOCATION_FREQUENCY_LOWER_BOUND < count as f64
                && count <= min(left_count, right_count)
            {
                let likelihood = crate::util::col_log_likelihood(
                    left_count as f64,
                    right_count as f64,
                    count as f64,
                    self.type_fdist.sum_counts() as f64,
                );

                if likelihood >= P::COLLOCATION_LOWER_BOUND
                    && (self.type_fdist.sum_counts() as f64 / left_count as f64)
                        > (right_count as f64 / count as f64)
                {
                    return Some((col, likelihood));
                }
            }
        }

        None
    }
}

struct PotentialSentenceStartersIterator<'b, I, P> {
    iter: I,
    sentence_break_count: usize,
    type_fdist: &'b FrequencyDistribution<&'b str>,
    sentence_starter_fdist: &'b FrequencyDistribution<&'b Token>,
    params: PhantomData<P>,
}

impl<'a, I, P> Iterator for PotentialSentenceStartersIterator<'_, I, P>
where
    I: Iterator<Item = &'a &'a Token>,
    P: TrainerParameters,
{
    type Item = (&'a Token, f64);

    #[inline]
    fn next(&mut self) -> Option<(&'a Token, f64)> {
        for tok in self.iter.by_ref() {
            let ss_count = self.sentence_starter_fdist.get(tok);
            let typ_count = self.type_fdist.get(tok.typ_with_period())
                + self.type_fdist.get(tok.typ_without_period());

            if typ_count < ss_count {
                continue;
            }

            let likelihood = crate::util::col_log_likelihood(
                self.sentence_break_count as f64,
                typ_count as f64,
                ss_count as f64,
                self.type_fdist.sum_counts() as f64,
            );

            let ratio = self.type_fdist.sum_counts() as f64 / self.sentence_break_count as f64;

            if likelihood >= P::SENTENCE_STARTER_LOWER_BOUND
                && ratio > (typ_count as f64 / ss_count as f64)
            {
                return Some((*tok, likelihood));
            }
        }

        None
    }
}

struct ConsecutiveItemIterator<'a, T: 'a, I>
where
    I: Iterator<Item = &'a T>,
{
    iter: I,
    last: Option<&'a T>,
}

impl<'a, T: 'a, I> Iterator for ConsecutiveItemIterator<'a, T, I>
where
    I: Iterator<Item = &'a T>,
{
    type Item = (&'a T, Option<&'a T>);

    #[inline]
    fn next(&mut self) -> Option<(&'a T, Option<&'a T>)> {
        match self.last {
            Some(i) => {
                self.last = self.iter.next();
                Some((i, self.last))
            }
            None => match self.iter.next() {
                Some(i) => {
                    self.last = self.iter.next();
                    Some((i, self.last))
                }
                None => None,
            },
        }
    }
}

// Macro for generating functions to load precompiled data.
macro_rules! preloaded_data(
    ($lang:ident, $file:expr) => (
        impl TrainingData {
            #[inline]
            /// Load precompiled data for a specific language
            pub fn $lang() -> TrainingData {
                FromStr::from_str(include_str!($file)).unwrap()
            }
        }
    )
);

preloaded_data!(czech, "data/czech.json");
preloaded_data!(danish, "data/danish.json");
preloaded_data!(dutch, "data/dutch.json");
preloaded_data!(english, "data/english.json");
preloaded_data!(estonian, "data/estonian.json");
preloaded_data!(finnish, "data/finnish.json");
preloaded_data!(french, "data/french.json");
preloaded_data!(german, "data/german.json");
preloaded_data!(greek, "data/greek.json");
preloaded_data!(italian, "data/italian.json");
preloaded_data!(norwegian, "data/norwegian.json");
preloaded_data!(polish, "data/polish.json");
preloaded_data!(portuguese, "data/portuguese.json");
preloaded_data!(slovene, "data/slovene.json");
preloaded_data!(spanish, "data/spanish.json");
preloaded_data!(swedish, "data/swedish.json");
preloaded_data!(turkish, "data/turkish.json");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_load_from_json_test() {
        let data: TrainingData = TrainingData::english();

        assert!(!data.orthographic_context.is_empty());
        assert!(!data.abbrevs.is_empty());
        assert!(!data.sentence_starters.is_empty());
        assert!(!data.collocations.is_empty());
        assert!(data.contains_sentence_starter("among"));
        assert!(data.contains_abbrev("w.va"));
        assert!(data.contains_collocation("##number##", "corrections"));
    }
}
