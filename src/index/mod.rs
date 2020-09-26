use crate::{common::*, dim::DynDim};

use base::*;

mod base {
    use super::*;

    pub trait DimIndex {}
    impl DimIndex for DynDim {}
    impl DimIndex for UTerm {}
    impl<U, B> DimIndex for UInt<U, B>
    where
        U: Unsigned,
        B: Bit,
    {
    }
    impl DimIndex for Z0 {}
    impl<U: Unsigned + NonZero> DimIndex for PInt<U> {}
    impl<U: Unsigned + NonZero> DimIndex for NInt<U> {}
}
