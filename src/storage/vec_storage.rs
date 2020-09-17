use super::Storage;
use crate::{common::*, dim::Dimensions};

// vec storage

pub struct VecStorage<T, D>
where
    D: Dimensions,
{
    dimensions: D,
    data: Vec<T>,
}

impl<T, D> Storage<T, D> for VecStorage<T, D> where D: Dimensions {}
