use super::{impls, ops};
use crate::{common::*, dims_t, DimsT};

pub use dyn_dim::*;

/// Marks the list of dimensions.
pub trait Dimensions
where
    Self: Sized,
{
    fn new_dyn_dims(dims: Vec<usize>) -> DynDimensions {
        DynDimensions(dims)
    }

    fn new_dims1<P>(first: P) -> Dims1<P::Output>
    where
        P: impls::IntoDim,
    {
        dims_t![first.into_dim()]
    }

    fn new_dims2<P, Q>(first: P, second: Q) -> Dims2<P::Output, Q::Output>
    where
        P: impls::IntoDim,
        Q: impls::IntoDim,
    {
        dims_t![first.into_dim(), second.into_dim()]
    }

    fn new_dims3<P, Q, R>(first: P, second: Q, third: R) -> Dims3<P::Output, Q::Output, R::Output>
    where
        P: impls::IntoDim,
        Q: impls::IntoDim,
        R: impls::IntoDim,
    {
        dims_t![first.into_dim(), second.into_dim(), third.into_dim()]
    }

    fn len(&self) -> usize
    where
        (): impls::LenImpl<Self>,
    {
        <() as impls::LenImpl<Self>>::impl_len(self)
    }

    fn to_vec(&self) -> Vec<usize>
    where
        (): impls::ToVecImpl<Self>,
    {
        <() as impls::ToVecImpl<_>>::impl_to_vec(self)
    }

    fn matrix_dot<Rhs>(
        &self,
        rhs: &Rhs,
    ) -> impls::MatrixDotImplOp<Self, Rhs, ops::MatrixDotOp<Self, Rhs>>
    where
        Self: Sized,
        Rhs: Dimensions,
        (): ops::MatrixDot<Self, Rhs>
            + impls::MatrixDotImpl<Self, Rhs, ops::MatrixDotOp<Self, Rhs>>,
    {
        <() as impls::MatrixDotImpl<Self, Rhs, ops::MatrixDotOp<Self, Rhs>>>::impl_matrix_dot(
            self, rhs,
        )
    }
}

impl Dimensions for DynDimensions {}

impl<Head, Tail> Dimensions for Cons<Head, Tail>
where
    Head: Dim,
    Tail: DimsList,
{
}

impl Dimensions for Nil {}

pub trait DimsList
where
    Self: List + Dimensions,
{
}

impl<Head, Tail> DimsList for Cons<Head, Tail>
where
    Head: Dim,
    Tail: DimsList,
{
}

impl DimsList for Nil {}

pub trait StaticDimsList
where
    Self: DimsList,
{
}
impl<Head, Tail> StaticDimsList for Cons<Head, Tail>
where
    Head: Unsigned + Dim,
    Tail: List + StaticDimsList,
{
}

impl StaticDimsList for Nil {}

/// Marks a single dimension.
pub trait Dim {
    fn to_usize(&self) -> usize;
}

impl Dim for Dyn {
    fn to_usize(&self) -> usize {
        self.0
    }
}
impl Dim for UTerm {
    fn to_usize(&self) -> usize {
        Self::USIZE
    }
}
impl<U, B> Dim for UInt<U, B>
where
    U: Unsigned,
    B: Bit,
{
    fn to_usize(&self) -> usize {
        Self::USIZE
    }
}

/// The dimensions with runtime length.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DynDimensions(pub Vec<usize>);

// aliases

pub type Dims1<P> = DimsT![P];
pub type Dims2<P, Q> = DimsT![P, Q];
pub type Dims3<P, Q, R> = DimsT![P, Q, R];

mod dyn_dim {
    use super::*;

    /// Single dynamic dimension.
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Dyn(pub usize);

    // add

    impl Add<Dyn> for Dyn {
        type Output = Dyn;

        fn add(self, rhs: Dyn) -> Self::Output {
            Dyn(self.0 + rhs.0)
        }
    }

    impl<U, B> Add<Dyn> for UInt<U, B>
    where
        U: Unsigned,
        B: Bit,
    {
        type Output = Dyn;

        fn add(self, rhs: Dyn) -> Self::Output {
            Dyn(Self::USIZE + rhs.0)
        }
    }

    impl<U, B> Add<UInt<U, B>> for Dyn
    where
        U: Unsigned,
        B: Bit,
    {
        type Output = Dyn;

        fn add(self, _rhs: UInt<U, B>) -> Self::Output {
            Dyn(self.0 + UInt::<U, B>::USIZE)
        }
    }

    impl Add<Dyn> for UTerm {
        type Output = Dyn;

        fn add(self, rhs: Dyn) -> Self::Output {
            rhs
        }
    }

    impl Add<UTerm> for Dyn {
        type Output = Dyn;

        fn add(self, _rhs: UTerm) -> Self::Output {
            self
        }
    }

    // sub

    impl Sub<Dyn> for Dyn {
        type Output = Dyn;

        fn sub(self, rhs: Dyn) -> Self::Output {
            Dyn(self.0 - rhs.0)
        }
    }

    impl<U, B> Sub<Dyn> for UInt<U, B>
    where
        U: Unsigned,
        B: Bit,
    {
        type Output = Dyn;

        fn sub(self, rhs: Dyn) -> Self::Output {
            Dyn(Self::USIZE - rhs.0)
        }
    }

    impl<U, B> Sub<UInt<U, B>> for Dyn
    where
        U: Unsigned,
        B: Bit,
    {
        type Output = Dyn;

        fn sub(self, _rhs: UInt<U, B>) -> Self::Output {
            Dyn(self.0 - UInt::<U, B>::USIZE)
        }
    }

    impl Sub<Dyn> for UTerm {
        type Output = Dyn;

        fn sub(self, rhs: Dyn) -> Self::Output {
            Dyn(0 - rhs.0)
        }
    }

    impl Sub<UTerm> for Dyn {
        type Output = Dyn;

        fn sub(self, _rhs: UTerm) -> Self::Output {
            self
        }
    }

    // mul

    impl Mul<Dyn> for Dyn {
        type Output = Dyn;

        fn mul(self, rhs: Dyn) -> Self::Output {
            Dyn(self.0 * rhs.0)
        }
    }

    impl<U, B> Mul<Dyn> for UInt<U, B>
    where
        U: Unsigned,
        B: Bit,
    {
        type Output = Dyn;

        fn mul(self, rhs: Dyn) -> Self::Output {
            Dyn(Self::USIZE * rhs.0)
        }
    }

    impl<U, B> Mul<UInt<U, B>> for Dyn
    where
        U: Unsigned,
        B: Bit,
    {
        type Output = Dyn;

        fn mul(self, _rhs: UInt<U, B>) -> Self::Output {
            Dyn(self.0 * UInt::<U, B>::USIZE)
        }
    }

    impl Mul<Dyn> for UTerm {
        type Output = UTerm;

        fn mul(self, _rhs: Dyn) -> Self::Output {
            UTerm
        }
    }

    impl Mul<UTerm> for Dyn {
        type Output = UTerm;

        fn mul(self, rhs: UTerm) -> Self::Output {
            rhs
        }
    }

    // div

    impl Div<Dyn> for Dyn {
        type Output = Dyn;

        fn div(self, rhs: Dyn) -> Self::Output {
            Dyn(self.0 / rhs.0)
        }
    }

    impl<U, B> Div<Dyn> for UInt<U, B>
    where
        U: Unsigned,
        B: Bit,
    {
        type Output = Dyn;

        fn div(self, rhs: Dyn) -> Self::Output {
            Dyn(Self::USIZE - rhs.0)
        }
    }

    impl<U, B> Div<UInt<U, B>> for Dyn
    where
        U: Unsigned,
        B: Bit,
    {
        type Output = Dyn;

        fn div(self, _rhs: UInt<U, B>) -> Self::Output {
            Dyn(self.0 / UInt::<U, B>::USIZE)
        }
    }

    impl Div<Dyn> for UTerm {
        type Output = Dyn;

        fn div(self, rhs: Dyn) -> Self::Output {
            Dyn(0 / rhs.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dims;

    #[test]
    fn len_test() {
        assert_eq!(dims![].len(), 0);
        assert_eq!(dims![3].len(), 1);
        assert_eq!(dims![1, 2, 3].len(), 3);
        assert_eq!(DynDimensions(vec![1, 2, 3]).len(), 3);
        assert_eq!(dims![2, 3].matrix_dot(&dims![3, 5]), dims![2, 5]);
    }
}
