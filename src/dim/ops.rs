use super::{Dim, Dimensions, Dims2, DimsList, Dyn, DynDimensions};
use crate::common::*;
use typenum::U1;

typ! {
    pub fn PushFront<dims, dim>(dims: Dimensions, dim: Dim) -> Dimensions {
        match dims {
            DynDimensions => DynDimensions,
            #[generics(head: Dim, tail: DimsList)]
            Cons::<head, tail> => {
                Cons::<dim, Cons<head, tail>>
            }
            Nil => {
                Cons::<dim, Nil>
            }
        }
    }

    pub fn PushBack<dims, dim>(dims: Dimensions, dim: Dim) -> Dimensions {
        match dims {
            DynDimensions => DynDimensions,
            #[generics(head: Dim, tail: DimsList)]
            Cons::<head, tail> => {
                let new_tail: List = PushBack(tail, dim);
                Cons::<head, new_tail>
            }
            Nil => {
                Cons::<dim, Nil>
            }
        }
    }

    // buggy
    // pub fn TensorDot<lhs, rhs, ndims>(lhs: Dimensions, rhs: Dimensions, ndims: Dim) -> Dimensions {
    //     if IsDynDimensions(lhs) || IsDynDimensions(rhs) || IsDyn(ndims) {
    //         DynDimensions
    //     } else {
    //         let ndims: Unsigned = ndims;
    //         let llen = Len(lhs);
    //         let rlen = Len(rhs);

    //         let lcontracted: DimsList = {
    //             let from: Unsigned = llen - ndims;
    //             Index(lhs, RangeFrom::<from>)
    //         };
    //         let rconstracted: DimsList = Index(rhs, RangeTo::<ndims>);

    //         let lremaining: DimsList = {
    //             let to: Unsigned = llen - ndims;
    //             Index(lhs, RangeTo::<to>)
    //         };
    //         let rremaining: DimsList = Index(rhs, RangeFrom::<ndims>);

    //         match (lcontracted, rcontracted) {
    //             #[capture(lcontracted)]
    //             (lcontracted, lcontracted) => {
    //                 Nil
    //                 // Extend(lremaiing, rremaining)
    //             }
    //         }
    //     }
    // }

    pub fn MatrixTranspose<dims>(dims: Dimensions) -> Dimensions {
        match dims {
            DynDimensions => DynDimensions,
            #[generics(p: Dim, q: Dim)]
            Dims2::<p, q> => Dims2::<q, p>,
        }
    }

    pub fn MatrixDot<lhs, rhs>(lhs: Dimensions, rhs: Dimensions) -> Dimensions {
        match (lhs, rhs) {
            (DynDimensions, DynDimensions) => Dims2::<Dyn, Dyn>,
            #[generics(q: Dim, r: Dim)]
            (DynDimensions, Dims2::<q, r>) => Dims2::<Dyn, r>,
            #[generics(p: Dim, q: Dim)]
            (Dims2::<p, q>, DynDimensions) => Dims2::<p, Dyn>,
            #[generics(p: Dim, r: Dim, uint: Unsigned, bit: Bit)]
            (Dims2::<p, UInt<uint, bit>>, Dims2::<UInt<uint, bit>, r>) => Dims2::<p, r>,
            #[generics(p: Dim, r: Dim, uint: Unsigned, bit: Bit)]
            (Dims2::<p, Dyn>, Dims2::<UInt<uint, bit>, r>) => Dims2::<p, r>,
            #[generics(p: Dim, r: Dim, uint: Unsigned, bit: Bit)]
            (Dims2::<p, UInt<uint, bit>>, Dims2::<Dyn, r>) => Dims2::<p, r>,
            #[generics(p: Dim, r: Dim)]
            (Dims2::<p, Dyn>, Dims2::<Dyn, r>) => Dims2::<p, r>,
        }
    }

    pub fn Flatten<input, start, end>(input: Dimensions, start: Dim, end: Dim) -> Dimensions {
        if IsDynDimensions(input) || IsDyn(start) || IsDyn(end) {
            DynDimensions
        } else {
            let input: DimsList = input;
            let start: Unsigned = start;
            let end: Unsigned = end;

            let heading: DimsList = Index(input, RangeTo::<start>);
            let trailing: DimsList = if end + 1u == Len(input) {
                Nil
            } else {
                let end_plus_1: Unsigned = end + 1u;
                Index(input, RangeFrom::<end_plus_1>)
            };
            let contracted: DimsList = Index(input, RangeInclusive::<(start, end)>);
            let product: Dim = ReduceProduct(contracted);
            Extend(heading, Cons::<product, trailing>)
        }
    }

    pub fn IsDynDimensions<dims>(dims: Dimensions) -> Bit {
        match dims {
            DynDimensions => true,
            #[generics(head, tail: DimsList)]
            Cons::<head, tail> => false,
            Nil => false,
        }
    }

    pub fn IsDyn<dim>(dim: Dim) -> Bit {
        match dim {
            Dyn => true,
            UTerm => false,
            #[generics(uint: Unsigned, bit: Bit)]
            UInt::<uint, bit> => false,
        }
    }
}

typ! {
    pub fn Combine<lhs, rhs>(lhs: Dimensions, rhs: Dimensions) -> Dimensions {
        if IsDynDimensions(lhs) || IsDynDimensions(rhs) {
            DynDimensions
        } else {
            let lhs: DimsList = lhs;
            let rhs: DimsList = rhs;
            CombineRecursive(lhs, rhs)
        }
    }

    pub fn CombineRecursive<lhs, rhs>(lhs: DimsList, rhs: DimsList) -> DimsList {
        match (lhs, rhs) {
            #[generics(ldim: Dim, ltail: DimsList, rdim: Dim, rtail: DimsList)]
            (Cons::<ldim, ltail>, Cons::<rdim, rtail>) => {
                let new_dim = if IsDyn(ldim) {
                    if IsDyn(rdim) {
                        Dyn
                    } else {
                        rdim
                    }
                } else if IsDyn(rdim) {
                    ldim
                } else {
                    match (ldim, rdim) {
                        #[capture(ldim)]
                        (ldim, ldim) => ldim
                    }
                };
                let new_tail = CombineRecursive(ltail, rtail);
                Cons::<new_dim, new_tail>
            }
            (Nil, Nil) => Nil
        }
    }
}

typ! {
    pub fn PyTorchBroadcast<lhs, rhs>(lhs: Dimensions, rhs: Dimensions) -> Dimensions {
        match (lhs, rhs) {
            #[capture(rhs)]
            (DynDimensions, rhs) => DynDimensions,
            #[generics(dim: Dim, tail: DimsList)]
            (Cons::<dim, tail>, DynDimensions) => DynDimensions,
            #[generics(ldim: Dim, ltail: DimsList, rdim: Dim, rtail: DimsList)]
            (Cons::<ldim, ltail>, Cons::<rdim, rtail>) => {
                let lhs: DimsList = lhs;
                let rhs: DimsList = rhs;

                let lhs_rev: DimsList = Reverse(lhs);
                let rhs_rev: DimsList = Reverse(rhs);

                Reverse(PyTorchBroadcastRecursive(lhs_rev, rhs_rev))
            }
        }
    }

    fn PyTorchBroadcastRecursive<lhs, rhs>(lhs: DimsList, rhs: DimsList) -> DimsList
    {
        match (lhs, rhs) {
            (Nil, Nil) => Nil,
            #[generics(dim: Dim, tail: DimsList)]
            (Nil, Cons::<dim, tail>) => rhs,
            #[generics(dim: Dim, tail: DimsList)]
            (Cons::<dim, tail>, Nil) => lhs,
            #[generics(ldim: Dim, ltail: DimsList, rdim: Dim, rtail: DimsList)]
            (Cons::<ldim, ltail>, Cons::<rdim, rtail>) => {
                let dim: Dim = match (ldim, rdim) {
                    #[capture(rdim)]
                    (Dyn, rdim) => Dyn,
                    #[generics(uint: Unsigned, bit: Bit)]
                    (UInt::<uint, bit>, Dyn) => Dyn,
                    #[generics(uint: Unsigned, bit: Bit)]
                    (U1, UInt::<uint, bit>) => rdim,
                    #[generics(uint: Unsigned, bit1: Unsigned, bit2: Bit)]
                    (UInt::<UInt<uint, bit1>, bit2>, U1) => ldim,
                    #[generics(uint: Unsigned, bit1: Unsigned, bit2: Bit)]
                    (UInt::<UInt<uint, bit1>, bit2>, UInt::<UInt<uint, bit1>, bit2>) => ldim,
                };
                let tail: DimsList = PyTorchBroadcastRecursive(ltail, rtail);
                Cons::<dim, tail>
            }
        }
    }
}

typ! {
    pub fn ConvDim<size, padding, dilation, ksize, stride>(size: Dim, padding: Dim, dilation: Dim, ksize: Dim, stride: Dim) -> Dim {
        let lhs: Dim = size + 2u * padding - dilation * (ksize - 1u) - 1u;
        DimIntegerDiv(lhs, stride) + 1u
    }

    pub fn ConvDimensions<sizes, paddings, dilations, ksizes, strides>(sizes: Dimensions, paddings: Dimensions, dilations: Dimensions, ksizes: Dimensions, strides: Dimensions) -> Dimensions {
        let is_dyn = IsDynDimensions(sizes)
            || IsDynDimensions(paddings)
            || IsDynDimensions(dilations)
            || IsDynDimensions(ksizes)
            || IsDynDimensions(strides);

        if is_dyn {
            DynDimensions
        } else {
            let sizes: DimsList = sizes;
            let paddings: DimsList = paddings;
            let dilations: DimsList = dilations;
            let ksizes: DimsList = ksizes;
            let strides: DimsList = strides;
            ConvDimsListRecursive(Nil, sizes, paddings, dilations, ksizes, strides)
        }
    }

    fn DimIntegerDiv<lhs, rhs>(lhs: Dim, rhs: Dim) -> Dim {
        if IsDyn(lhs) || IsDyn(rhs) {
            Dyn
        } else {
            let lhs: Unsigned = lhs;
            let rhs: Unsigned = rhs;
            (lhs - (lhs % rhs)) / rhs
        }
    }

    fn ConvDimsListRecursive<saved, sizes, paddings, dilations, ksizes, strides>(saved: DimsList, sizes: DimsList, paddings: DimsList, dilations: DimsList, ksizes: DimsList, strides: DimsList) -> DimsList {
        match sizes {
            #[generics(head, tail: DimsList)]
            Cons::<head, tail> => {
                let size: Dim = First(sizes);
                let padding: Dim = First(paddings);
                let dilation: Dim = First(dilations);
                let ksize: Dim = First(ksizes);
                let stride: Dim = First(strides);

                let new_sizes: DimsList = PopFront(sizes);
                let new_paddings: DimsList = PopFront(paddings);
                let new_dilations: DimsList = PopFront(dilations);
                let new_ksizes: DimsList = PopFront(ksizes);
                let new_strides: DimsList = PopFront(strides);

                let dim: Dim = ConvDim(size, padding, dilation, ksize, stride);
                let new_saved: DimsList = Cons::<dim, saved>;
                ConvDimsListRecursive(new_saved, new_sizes, new_paddings, new_dilations, new_ksizes, new_strides)
            }
            Nil => Reverse(saved),
        }
    }
}

typ! {
    pub fn Cat<inputs, index>(inputs: List, index: Dim) -> Dimensions {
        if CheckDynDimensions(inputs) || IsDyn(index) {
            DynDimensions
        } else {
            let index: Unsigned = index;
            MergeLeadingDims(Nil, inputs, index)
        }
    }

    fn MergeLeadingDims<saved, remaining, index>(saved: DimsList, remaining: List, index: Unsigned) -> DimsList {
        if index == 0u {
            let new_dim = SumDims(remaining);
            let new_remaining = RemoveDims(Nil, remaining);
            let new_saved = Cons::<new_dim, saved>;
            MergeTrailingDims(new_saved, new_remaining)
        } else {
            let expect = ExtractFirstDim(Dyn, remaining);
            let new_remaining = RemoveExpectedDims(Nil, remaining, expect);
            let new_saved = Cons::<expect, saved>;
            let new_index: Unsigned = index - 1u;
            MergeLeadingDims(new_saved, new_remaining, new_index)
        }
    }

    fn MergeTrailingDims<saved, remaining>(saved: DimsList, remaining: List) -> DimsList {
        match remaining {
            #[generics(dim: Dim, dims_tail: DimsList, tail: List)]
            Cons::<Cons::<dim, dims_tail>, tail> => {
                let expect = ExtractFirstDim(Dyn, remaining);
                let new_remaining = RemoveExpectedDims(Nil, remaining, expect);
                let new_saved = Cons::<expect, saved>;
                MergeTrailingDims(new_saved, new_remaining)
            }
            #[generics(tail: List)]
            Cons::<Nil, tail> => {
                AssertAllEmpty(remaining);
                Reverse(saved)
            }
        }
    }

    fn AssertAllEmpty<remaining>(remaining: List) {
        match remaining {
            #[generics(tail: List)]
            Cons::<Nil, tail> => AssertAllEmpty(tail),
            Nil => (),
        }
    }

    fn CheckDynDimensions<remaining>(remaining: List) -> Bit {
        match remaining {
            #[generics(tail: List)]
            Cons::<DynDimensions, tail> => true,
            #[generics(dim: Dim, dims_tail: DimsList, tail: List)]
            Cons::<Cons::<dim, dims_tail>, tail> => CheckDynDimensions(tail),
            Nil => false,
        }
    }

    fn ExtractFirstDim<expect, remaining>(expect: Dim, remaining: List) -> Dim {
        match remaining {
            #[generics(dim: Dim, dims_tail: DimsList, tail: List)]
            Cons::<Cons<dim, dims_tail>, tail> => {
                match (expect, dim) {
                    #[capture(dim)]
                    (Dyn, dim) => ExtractFirstDim(dim, tail),
                    (UTerm, Dyn) => ExtractFirstDim(expect, tail),
                    #[generics(uint: Unsigned, bit: Bit)]
                    (UInt::<uint, bit>, Dyn) => ExtractFirstDim(expect, tail),
                    (UTerm, UTerm) => ExtractFirstDim(expect, tail),
                    #[generics(uint: Unsigned, bit: Bit)]
                    (UInt::<uint, bit>, UInt::<uint, bit>) => ExtractFirstDim(expect, tail),
                }
            }
            Nil => expect,
        }
    }

    fn RemoveDims<saved, remaining>(saved: List, remaining: List) -> List {
        match remaining {
            #[generics(dim: Dim, dims_tail: DimsList, tail: List)]
            Cons::<Cons::<dim, dims_tail>, tail> => RemoveDims(Cons::<dims_tail, saved>, tail),
            Nil => Reverse(saved),
        }
    }

    fn RemoveExpectedDims<saved, remaining, expect>(saved: List, remaining: List, expect: Dim) -> List {
        match remaining {
            #[generics(dim: Dim, dims_tail: DimsList, tail: List)]
            Cons::<Cons::<dim, dims_tail>, tail> => {
                match (expect, dim) {
                    #[capture(dim)]
                    (Dyn, Dyn) => RemoveExpectedDims(Cons::<dims_tail, saved>, tail, expect),
                    (UTerm, Dyn) => RemoveExpectedDims(Cons::<dims_tail, saved>, tail, expect),
                    #[generics(uint: Unsigned, bit: Bit)]
                    (UInt::<uint, bit>, Dyn) => RemoveExpectedDims(Cons::<dims_tail, saved>, tail, expect),
                    (UTerm, UTerm) => RemoveExpectedDims(Cons::<dims_tail, saved>, tail, expect),
                    #[generics(uint: Unsigned, bit: Bit)]
                    (UInt::<uint, bit>, UInt::<uint, bit>) => RemoveExpectedDims(Cons::<dims_tail, saved>, tail, expect),
                }
            }
            Nil => Reverse(saved),
        }
    }

    fn SumDims<remaining>(remaining: List) -> Dim {
        match remaining {
            #[generics(dim: Dim, dims_tail: DimsList, tail: List)]
            Cons::<Cons::<dim, dims_tail>, tail> => {
                let remaining_sum = SumDims(tail);
                match (remaining_sum, dim) {
                    #[capture(dim)]
                    (Dyn, dim) => Dyn,
                    (UTerm, Dyn) => Dyn,
                    (UTerm, UTerm) => UTerm,
                    #[generics(uint: Unsigned, bit: Bit)]
                    (UTerm, UInt::<uint, bit>) => dim,
                    #[generics(uint: Unsigned, bit: Bit)]
                    (UInt::<uint, bit>, Dyn) => Dyn,
                    #[generics(uint: Unsigned, bit: Bit)]
                    (UInt::<uint, bit>, UTerm) => remaining_sum,
                    #[generics(uint1: Unsigned, bit1: Bit, uint2: Unsigned, bit2: Bit)]
                    (UInt::<uint1, bit1>, UInt::<uint2, bit2>) => dim + remaining_sum,
                }
            }
            Nil => UTerm,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dims;
    use typenum::consts::*;

    #[test]
    fn test() {
        let _: SameOp<CombineOp<Dims![?], Dims![2, 3, 4]>, Dims![?]> = ();
        let _: SameOp<CombineOp<Dims![2, 3, 4], Dims![?]>, Dims![?]> = ();
        let _: SameOp<CombineOp<Dims![2, 3, 4], Dims![2, 3, 4]>, Dims![2, 3, 4]> = ();
        let _: SameOp<CombineOp<Dims![_, _, 4], Dims![_, 3, _]>, Dims![_, 3, 4]> = ();
        let _: SameOp<MatrixTransposeOp<Dims![?]>, Dims![?]> = ();
        let _: SameOp<MatrixTransposeOp<Dims![2, _]>, Dims![_, 2]> = ();
        let _: SameOp<MatrixTransposeOp<Dims![2, 3]>, Dims![3, 2]> = ();
        let _: SameOp<MatrixDotOp<Dims![?], Dims![?]>, Dims![_, _]> = ();
        let _: SameOp<MatrixDotOp<Dims![?], Dims![3, 5]>, Dims![_, 5]> = ();
        let _: SameOp<MatrixDotOp<Dims![2, 3], Dims![?]>, Dims![2, _]> = ();
        let _: SameOp<MatrixDotOp<Dims![2, 3], Dims![3, 5]>, Dims![2, 5]> = ();
        let _: SameOp<MatrixDotOp<Dims![2, 3], Dims![_, 5]>, Dims![2, 5]> = ();
        let _: SameOp<MatrixDotOp<Dims![2, _], Dims![3, 5]>, Dims![2, 5]> = ();
        let _: SameOp<MatrixDotOp<Dims![2, _], Dims![_, 5]>, Dims![2, 5]> = ();
        let _: SameOp<MatrixDotOp<Dims![_, 3], Dims![_, 5]>, Dims![_, 5]> = ();
        let _: SameOp<FlattenOp<Dims![1, 2, 3], U0, U2>, Dims![6]> = ();
        let _: SameOp<FlattenOp<Dims![1, 2, 3], U1, U2>, Dims![1, 6]> = ();
        let _: SameOp<FlattenOp<Dims![1, 2, 3], U0, U1>, Dims![2, 3]> = ();
        let _: SameOp<FlattenOp<Dims![1, 2, 3], U1, U1>, Dims![1, 2, 3]> = ();
        let _: SameOp<FlattenOp<Dims![1, _, 3], U0, U1>, Dims![_, 3]> = ();
        let _: SameOp<CatOp<List![Dims![1, 2, 3], Dims![?]], tyuint!(1)>, Dims![?]> = ();
        let _: SameOp<CatOp<List![Dims![1, 2, 3], Dims![1, 5, 3]], Dyn>, Dims![?]> = ();
        let _: SameOp<CatOp<List![Dims![2], Dims![3]], tyuint!(0)>, Dims![5]> = ();
        let _: SameOp<CatOp<List![Dims![2], Dims![_]], tyuint!(0)>, Dims![_]> = ();
        let _: SameOp<CatOp<List![Dims![_], Dims![_]], tyuint!(0)>, Dims![_]> = ();
        let _: SameOp<CatOp<List![Dims![2, 5, 3], Dims![2, 1, 3]], tyuint!(1)>, Dims![2, 6, 3]> =
            ();
        let _: SameOp<
            CatOp<List![Dims![_, 5, _, 3], Dims![2, 1, _, _]], tyuint!(1)>,
            Dims![2, 6, _, 3],
        > = ();
        let _: SameOp<
            CatOp<List![Dims![_, 7, _, 3], Dims![2, 7, 8, _]], tyuint!(2)>,
            Dims![2, 7, _, 3],
        > = ();
        let _: SameOp<
            CatOp<List![Dims![_, 7, 1, 3], Dims![2, _, 8, _], Dims![2, 7, 4, _]], tyuint!(2)>,
            Dims![2, 7, 13, 3],
        > = ();
    }
}
