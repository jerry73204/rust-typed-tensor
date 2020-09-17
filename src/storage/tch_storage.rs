use super::Storage;
use crate::{
    common::*,
    dim::{Dim, Dimensions},
};

pub use device::*;

// tch storage

pub struct TchStorage<T, D, Dev>
where
    T: tch::kind::Element,
    D: Dimensions,
    Dev: Device,
{
    device: Dev,
    data: tch::Tensor,
    _phantom: PhantomData<(T, D)>,
}

impl<T, D, Dev> Storage<T, D> for TchStorage<T, D, Dev>
where
    D: Dimensions,
    T: tch::kind::Element,
    Dev: Device,
{
}

mod device {
    use super::*;

    pub trait Device {}

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct Cpu;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct Gpu<Index>(pub Index)
    where
        Index: Dim;
}
