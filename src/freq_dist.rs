// Copyright 2016 rust-punkt developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Index;

/// A frequency distribution that keeps track of the number of times
/// each item is seen.
#[derive(Debug, Clone)]
pub struct FrequencyDistribution<T>
where
  T: Eq + Hash,
{
  counts: HashMap<T, usize>,
  total_count: usize,
}

impl<T> FrequencyDistribution<T>
where
  T: Eq + Hash,
{
  /// Creates a new empty frequency distribution.
  pub fn new() -> Self {
    FrequencyDistribution {
      counts: HashMap::new(),
      total_count: 0,
    }
  }

  /// Adds an item to the frequency distribution, incrementing its count.
  /// Returns the new count of the item.
  pub fn insert<U>(&mut self, item: U) -> usize
  where
    U: Into<T>,
  {
    let item = item.into();
    let count = self.counts.entry(item).or_insert(0);
    *count += 1;
    self.total_count += 1;
    *count
  }

  /// Gets the frequency of an item.
  pub fn get<Q>(&self, item: &Q) -> usize
  where
    T: std::borrow::Borrow<Q>,
    Q: Hash + Eq + ?Sized,
  {
    self.counts.get(item).copied().unwrap_or(0)
  }

  /// Gets the total number of items counted (sum of all frequencies).
  pub fn sum_counts(&self) -> usize {
    self.total_count
  }

  /// Returns an iterator over the keys (items) in the distribution.
  pub fn keys(&self) -> impl Iterator<Item = &T> {
    self.counts.keys()
  }
}

impl<T> Default for FrequencyDistribution<T>
where
  T: Eq + Hash,
{
  fn default() -> Self {
    Self::new()
  }
}

impl<T, Q> Index<&Q> for FrequencyDistribution<T>
where
  T: Eq + Hash + std::borrow::Borrow<Q>,
  Q: Hash + Eq + ?Sized,
{
  type Output = usize;

  fn index(&self, key: &Q) -> &Self::Output {
    self.counts.get(key).unwrap_or(&0)
  }
}
