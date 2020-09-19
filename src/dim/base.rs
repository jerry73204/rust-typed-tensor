use super::{impls, ops};
use crate::{common::*, dims_t, DimsT};

pub use dim::*;
pub use dimensions::*;

mod dimensions {
    use super::*;

    // dynamic type

    /// The dimensions with runtime length.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct DynDimensions(pub Vec<usize>);

    // aliases

    pub type Dims1<P> = DimsT![P];
    pub type Dims2<P, Q> = DimsT![P, Q];
    pub type Dims3<P, Q, R> = DimsT![P, Q, R];

    /// Marks the list of dimensions.
    pub trait Dimensions
    where
        Self: Sized,
    {
        fn new<Shape>(shape: Shape) -> Self
        where
            (): impls::NewDimsImpl<Shape, Self>,
        {
            <() as impls::NewDimsImpl<Shape, Self>>::impl_new(shape)
        }

        fn to_dyn(&self) -> DynDimensions
        where
            (): impls::ToDynImpl<Self>,
        {
            <() as impls::ToDynImpl<Self>>::impl_to_dyn(self)
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
}

mod dim {
    use super::*;

    // alias

    pub type DynDim = Dyn<usize>;

    // trait

    /// Marks a single dimension.
    pub trait Dim {
        fn to_usize(&self) -> usize;
        fn to_dyn(&self) -> DynDim;
    }

    impl Dim for DynDim {
        fn to_usize(&self) -> usize {
            self.0
        }

        fn to_dyn(&self) -> DynDim {
            self.clone()
        }
    }
    impl Dim for UTerm {
        fn to_usize(&self) -> usize {
            Self::USIZE
        }

        fn to_dyn(&self) -> DynDim {
            DynDim::new(Self::USIZE)
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

        fn to_dyn(&self) -> DynDim {
            DynDim::new(Self::USIZE)
        }
    }
}

// dynamic dim

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dims, Dims};
    use typenum::consts::*;

    #[test]
    fn len_test() {
        assert_eq!(<Dims![?]>::new(vec![3, 1, 4]), DynDimensions(vec![3, 1, 4]));
        assert_eq!(
            <Dims![?]>::new(&vec![3, 1, 4]),
            DynDimensions(vec![3, 1, 4])
        );
        assert_eq!(<Dims![_]>::new(3), dims_t![DynDim::new(3)]);
        assert_eq!(<Dims![3]>::new(U3::new()), dims![3]);
        assert_eq!(
            <Dims![_, 3]>::new((2, U3::new())),
            dims_t![DynDim::new(2), U3::new()]
        );
        assert_eq!(
            <Dims![4, _, 3]>::new((U4::new(), 2, U3::new())),
            dims_t![U4::new(), DynDim::new(2), U3::new()]
        );
        assert_eq!(dims![].len(), 0);
        assert_eq!(dims![3].len(), 1);
        assert_eq!(dims![1, 2, 3].len(), 3);
        assert_eq!(DynDimensions(vec![1, 2, 3]).len(), 3);
        assert_eq!(dims![2, 3].matrix_dot(&dims![3, 5]), dims![2, 5]);
    }
}
