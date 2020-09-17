use super::*;
use crate::{common::*, dims_t, DimsT};

// len

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

// to vec

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

// matrix dot

pub trait MatrixDotImpl<Lhs, Rhs, Out>
where
    Lhs: Dimensions,
    Rhs: Dimensions,
    Out: Dimensions,
{
    fn impl_matrix_dot(lhs: &Lhs, rhs: &Rhs) -> Out;
}

impl<Lhs, Rhs> MatrixDotImpl<Lhs, Rhs, DynDimensions> for ()
where
    Lhs: Dimensions,
    Rhs: Dimensions,
    (): ToVecImpl<Lhs>,
    (): ToVecImpl<Rhs>,
{
    fn impl_matrix_dot(lhs: &Lhs, rhs: &Rhs) -> DynDimensions {
        let ldims = <()>::impl_to_vec(lhs);
        let rdims = <()>::impl_to_vec(rhs);

        // TODO: return Result instead
        assert!(
            ldims.len() == 2,
            "the left hand side dimension size is not 2"
        );
        assert!(
            rdims.len() == 2,
            "the right hand side dimension size is not 2"
        );
        assert!(ldims[1] == rdims[0], "contracted dimension mismatch");
        assert!(ldims[1] != 0, "the contracted dimension must not be zero");

        DynDimensions(vec![ldims[0], rdims[1]])
    }
}

impl<P, R, U, B> MatrixDotImpl<DimsT![P, UInt::<U, B>], DimsT![UInt::<U, B>, R], DimsT![P, R]>
    for ()
where
    P: Dim + Clone,
    R: Dim + Clone,
    U: Unsigned,
    B: Bit,
{
    fn impl_matrix_dot(
        lhs: &DimsT![P, UInt::<U, B>],
        rhs: &DimsT![UInt::<U, B>, R],
    ) -> DimsT![P, R] {
        dims_t![lhs.head.clone(), rhs.tail.head.clone()]
    }
}

impl<P, R, U, B> MatrixDotImpl<DimsT![P, Dyn], DimsT![UInt::<U, B>, R], DimsT![P, R]> for ()
where
    P: Dim + Clone,
    R: Dim + Clone,
    U: Unsigned,
    B: Bit,
{
    fn impl_matrix_dot(lhs: &DimsT![P, Dyn], rhs: &DimsT![UInt::<U, B>, R]) -> DimsT![P, R] {
        assert!(
            lhs.tail.head.0 == UInt::<U, B>::USIZE,
            "dimensions mismatch"
        );
        dims_t![lhs.head.clone(), rhs.tail.head.clone()]
    }
}

impl<P, R, U, B> MatrixDotImpl<DimsT![P, UInt::<U, B>], DimsT![Dyn, R], DimsT![P, R]> for ()
where
    P: Dim + Clone,
    R: Dim + Clone,
    U: Unsigned,
    B: Bit,
{
    fn impl_matrix_dot(lhs: &DimsT![P, UInt::<U, B>], rhs: &DimsT![Dyn, R]) -> DimsT![P, R] {
        assert!(rhs.head.0 == UInt::<U, B>::USIZE, "dimensions mismatch");
        dims_t![lhs.head.clone(), rhs.tail.head.clone()]
    }
}
