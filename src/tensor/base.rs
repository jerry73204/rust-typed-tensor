use crate::{
    common::*,
    dim::{Dimensions, Dims2},
    storage::Storage,
};
use typenum::U1;

pub struct Tensor<T, D, S>
where
    D: Dimensions,
    S: Storage<T, D>,
{
    data: S,
    _phantom: PhantomData<(T, D)>,
}

// aliases

pub type Vector<T, D, S> = Matrix<T, D, U1, S>;
pub type Matrix<T, R, C, S> = Tensor<T, Dims2<R, C>, S>;
