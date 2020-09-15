use crate::{common::*, dim::Dimensions};

// traits

pub trait Storage<T, D>
where
    D: Dimensions,
{
}

// vec storage

pub struct VecStorage<T, D>
where
    D: Dimensions,
{
    dimensions: D,
    data: Vec<T>,
}

impl<T, D> Storage<T, D> for VecStorage<T, D> where D: Dimensions {}

// tch storage

pub struct TchStorage<T, D>
where
    T: tch::kind::Element,
    D: Dimensions,
{
    data: tch::Tensor,
    _phantom: PhantomData<(T, D)>,
}

impl<T, D> Storage<T, D> for TchStorage<T, D>
where
    D: Dimensions,
    T: tch::kind::Element,
{
}
