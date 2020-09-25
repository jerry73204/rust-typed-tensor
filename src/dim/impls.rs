use super::{ops, *};
use crate::{common::*, dims_t, DimsT};

// into dim

pub use into_dim::*;

mod into_dim {
    use super::*;

    pub trait IntoDim {
        type Output;

        fn into_dim(self) -> Self::Output;
    }

    impl IntoDim for usize {
        type Output = DynDim;

        fn into_dim(self) -> Self::Output {
            DynDim::new(self)
        }
    }

    impl IntoDim for UTerm {
        type Output = Self;

        fn into_dim(self) -> Self::Output {
            Self::new()
        }
    }

    impl<U, B> IntoDim for UInt<U, B>
    where
        U: Unsigned,
        B: Bit,
    {
        type Output = Self;

        fn into_dim(self) -> Self::Output {
            Self::new()
        }
    }
}

// new impl
pub use new::*;

mod new {
    use super::*;
    pub trait NewDimsImpl<Shape, Out> {
        fn impl_new(shape: Shape) -> Out;
    }

    impl NewDimsImpl<Vec<usize>, DynDimensions> for () {
        fn impl_new(shape: Vec<usize>) -> DynDimensions {
            DynDimensions(shape)
        }
    }

    impl NewDimsImpl<&Vec<usize>, DynDimensions> for () {
        fn impl_new(shape: &Vec<usize>) -> DynDimensions {
            DynDimensions(shape.to_owned())
        }
    }

    impl NewDimsImpl<&[usize], DynDimensions> for () {
        fn impl_new(shape: &[usize]) -> DynDimensions {
            DynDimensions(shape.iter().cloned().collect())
        }
    }

    impl<PFrom, PTo> NewDimsImpl<PFrom, DimsT![PTo]> for ()
    where
        PFrom: IntoDim<Output = PTo>,
    {
        fn impl_new(shape: PFrom) -> DimsT![PTo] {
            dims_t![shape.into_dim()]
        }
    }

    impl<PFrom, PTo> NewDimsImpl<(PFrom,), DimsT![PTo]> for ()
    where
        PFrom: IntoDim<Output = PTo>,
    {
        fn impl_new((shape,): (PFrom,)) -> DimsT![PTo] {
            dims_t![shape.into_dim()]
        }
    }

    impl<PFrom, PTo, QFrom, QTo> NewDimsImpl<(PFrom, QFrom), DimsT![PTo, QTo]> for ()
    where
        PFrom: IntoDim<Output = PTo>,
        QFrom: IntoDim<Output = QTo>,
    {
        fn impl_new((d0, d1): (PFrom, QFrom)) -> DimsT![PTo, QTo] {
            dims_t![d0.into_dim(), d1.into_dim()]
        }
    }

    impl<PFrom, PTo, QFrom, QTo, RFrom, RTo>
        NewDimsImpl<(PFrom, QFrom, RFrom), DimsT![PTo, QTo, RTo]> for ()
    where
        PFrom: IntoDim<Output = PTo>,
        QFrom: IntoDim<Output = QTo>,
        RFrom: IntoDim<Output = RTo>,
    {
        fn impl_new((d0, d1, d2): (PFrom, QFrom, RFrom)) -> DimsT![PTo, QTo, RTo] {
            dims_t![d0.into_dim(), d1.into_dim(), d2.into_dim()]
        }
    }
}

// to_dyn
pub use to_dyn::*;

mod to_dyn {
    use super::*;

    pub trait ToDynImpl<Dims>
    where
        Dims: Dimensions,
    {
        fn impl_to_dyn(dims: &Dims) -> DynDimensions;
        fn impl_to_dyn_recursive(dims: &Dims, values: &mut Vec<usize>);
    }

    impl ToDynImpl<DynDimensions> for () {
        fn impl_to_dyn(dims: &DynDimensions) -> DynDimensions {
            dims.clone()
        }

        fn impl_to_dyn_recursive(dims: &DynDimensions, values: &mut Vec<usize>) {
            unreachable!("please report bug");
        }
    }

    impl<Head, Tail> ToDynImpl<Cons<Head, Tail>> for ()
    where
        Head: Dim,
        Tail: DimsList,
        (): ToDynImpl<Tail>,
    {
        fn impl_to_dyn(dims: &Cons<Head, Tail>) -> DynDimensions {
            let mut values = vec![];
            <() as ToDynImpl<Cons<Head, Tail>>>::impl_to_dyn_recursive(dims, &mut values);
            DynDimensions(values)
        }

        fn impl_to_dyn_recursive(dims: &Cons<Head, Tail>, values: &mut Vec<usize>) {
            values.push(dims.head.to_dyn().0);
            <() as ToDynImpl<Tail>>::impl_to_dyn_recursive(&dims.tail, values);
        }
    }

    impl ToDynImpl<Nil> for () {
        fn impl_to_dyn(dims: &Nil) -> DynDimensions {
            DynDimensions(vec![])
        }

        fn impl_to_dyn_recursive(_dims: &Nil, _values: &mut Vec<usize>) {}
    }
}

// len
pub use len::*;

mod len {
    use super::*;

    pub trait LenImpl<Dims>
    where
        Dims: Dimensions,
    {
        fn impl_len(dims: &Dims) -> usize;
    }

    impl LenImpl<DynDimensions> for () {
        fn impl_len(dims: &DynDimensions) -> usize {
            dims.0.len()
        }
    }

    impl<Head, Tail> LenImpl<Cons<Head, Tail>> for ()
    where
        Head: Dim,
        Tail: DimsList,
        (): LenImpl<Tail>,
    {
        fn impl_len(dims: &Cons<Head, Tail>) -> usize {
            1 + <() as LenImpl<Tail>>::impl_len(&dims.tail)
        }
    }

    impl LenImpl<Nil> for () {
        fn impl_len(dims: &Nil) -> usize {
            0
        }
    }
}

// to vec
pub use to_vec::ToVecImpl;

mod to_vec {
    use super::*;

    pub trait ToVecImpl<Dims>
    where
        Dims: Dimensions,
    {
        fn impl_to_vec(dims: &Dims) -> Vec<usize>;
        fn impl_to_vec_recursive(dims: &Dims, dims: &mut Vec<usize>);
    }

    impl ToVecImpl<DynDimensions> for () {
        fn impl_to_vec(dims: &DynDimensions) -> Vec<usize> {
            dims.0.clone()
        }

        fn impl_to_vec_recursive(dims: &DynDimensions, _dims: &mut Vec<usize>) {
            unreachable!();
        }
    }

    impl<Head, Tail> ToVecImpl<Cons<Head, Tail>> for ()
    where
        Head: Dim,
        Tail: DimsList,
        (): ToVecImpl<Tail>,
    {
        fn impl_to_vec(dims: &Cons<Head, Tail>) -> Vec<usize> {
            let mut extended = vec![];
            <() as ToVecImpl<Cons<Head, Tail>>>::impl_to_vec_recursive(dims, &mut extended);
            extended
        }

        fn impl_to_vec_recursive(dims: &Cons<Head, Tail>, extended: &mut Vec<usize>) {
            extended.push(dims.head.to_usize());
            <() as ToVecImpl<Tail>>::impl_to_vec_recursive(&dims.tail, extended)
        }
    }

    impl ToVecImpl<Nil> for () {
        fn impl_to_vec(_dims: &Nil) -> Vec<usize> {
            vec![]
        }
        fn impl_to_vec_recursive(_dims: &Nil, _extended: &mut Vec<usize>) {}
    }
}

// matrix dot

pub use matrix_dot::{MatrixDotImpl, MatrixDotImplOp};

mod matrix_dot {
    use super::*;

    pub trait MatrixDotImpl<Lhs, Rhs>
    where
        Lhs: Dimensions,
        Rhs: Dimensions,
    {
        type Output;
        fn impl_matrix_dot(lhs: &Lhs, rhs: &Rhs) -> Self::Output;
    }

    impl<Lhs, Rhs> MatrixDotImpl<Lhs, Rhs> for ()
    where
        Lhs: Dimensions,
        Rhs: Dimensions,
        (): MatrixDotInnerImpl<Lhs, Rhs, ops::MatrixDotOp<Lhs, Rhs>> + ops::MatrixDot<Lhs, Rhs>,
    {
        type Output = MatrixDotInnerImplOp<Lhs, Rhs, ops::MatrixDotOp<Lhs, Rhs>>;

        fn impl_matrix_dot(lhs: &Lhs, rhs: &Rhs) -> Self::Output {
            <() as MatrixDotInnerImpl<Lhs, Rhs, ops::MatrixDotOp<Lhs, Rhs>>>::impl_matrix_dot_inner(
                lhs, rhs,
            )
        }
    }

    pub trait MatrixDotInnerImpl<Lhs, Rhs, Out>
    where
        Lhs: Dimensions,
        Rhs: Dimensions,
        Out: Dimensions,
    {
        type Output;
        fn impl_matrix_dot_inner(lhs: &Lhs, rhs: &Rhs) -> Self::Output;
    }

    impl<Lhs, Rhs> MatrixDotInnerImpl<Lhs, Rhs, DynDimensions> for ()
    where
        Lhs: Dimensions,
        Rhs: Dimensions,
        (): ToVecImpl<Lhs> + ToVecImpl<Rhs>,
    {
        type Output = Result<DynDimensions>;

        fn impl_matrix_dot_inner(lhs: &Lhs, rhs: &Rhs) -> Self::Output {
            let ldims = <()>::impl_to_vec(lhs);
            let rdims = <()>::impl_to_vec(rhs);

            // TODO: return Result instead
            ensure!(
                ldims.len() == 2,
                "the left hand side dimension size is not 2"
            );
            ensure!(
                rdims.len() == 2,
                "the right hand side dimension size is not 2"
            );
            ensure!(ldims[1] == rdims[0], "contracted dimension mismatch");
            ensure!(ldims[1] != 0, "the contracted dimension must not be zero");
            Ok(DynDimensions(vec![ldims[0], rdims[1]]))
        }
    }

    impl<P, R, U, B>
        MatrixDotInnerImpl<DimsT![P, UInt::<U, B>], DimsT![UInt::<U, B>, R], DimsT![P, R]> for ()
    where
        P: Dim + Clone,
        R: Dim + Clone,
        U: Unsigned,
        B: Bit,
    {
        type Output = DimsT![P, R];

        fn impl_matrix_dot_inner(
            lhs: &DimsT![P, UInt::<U, B>],
            rhs: &DimsT![UInt::<U, B>, R],
        ) -> Self::Output {
            dims_t![lhs.head.clone(), rhs.tail.head.clone()]
        }
    }

    impl<P, R, U, B> MatrixDotInnerImpl<DimsT![P, DynDim], DimsT![UInt::<U, B>, R], DimsT![P, R]> for ()
    where
        P: Dim + Clone,
        R: Dim + Clone,
        U: Unsigned,
        B: Bit,
    {
        type Output = Result<DimsT![P, R]>;

        fn impl_matrix_dot_inner(
            lhs: &DimsT![P, DynDim],
            rhs: &DimsT![UInt::<U, B>, R],
        ) -> Self::Output {
            ensure!(
                lhs.tail.head.0 == UInt::<U, B>::USIZE,
                "dimensions mismatch"
            );
            Ok(dims_t![lhs.head.clone(), rhs.tail.head.clone()])
        }
    }

    impl<P, R, U, B> MatrixDotInnerImpl<DimsT![P, UInt::<U, B>], DimsT![DynDim, R], DimsT![P, R]> for ()
    where
        P: Dim + Clone,
        R: Dim + Clone,
        U: Unsigned,
        B: Bit,
    {
        type Output = Result<DimsT![P, R]>;

        fn impl_matrix_dot_inner(
            lhs: &DimsT![P, UInt::<U, B>],
            rhs: &DimsT![DynDim, R],
        ) -> Self::Output {
            ensure!(rhs.head.0 == UInt::<U, B>::USIZE, "dimensions mismatch");
            Ok(dims_t![lhs.head.clone(), rhs.tail.head.clone()])
        }
    }

    pub type MatrixDotImplOp<Lhs, Rhs> = <() as MatrixDotImpl<Lhs, Rhs>>::Output;
    type MatrixDotInnerImplOp<Lhs, Rhs, Out> = <() as MatrixDotInnerImpl<Lhs, Rhs, Out>>::Output;
}
