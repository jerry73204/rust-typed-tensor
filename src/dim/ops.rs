use super::{Dim, Dimensions, Dims2, DimsList, DynDim, DynDimensions, StaticDimsList};
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
            (DynDimensions, DynDimensions) => Dims2::<DynDim, DynDim>,
            #[generics(q: Dim, r: Dim)]
            (DynDimensions, Dims2::<q, r>) => Dims2::<DynDim, r>,
            #[generics(p: Dim, q: Dim)]
            (Dims2::<p, q>, DynDimensions) => Dims2::<p, DynDim>,
            #[generics(p: Dim, r: Dim, uint: Unsigned, bit: Bit)]
            (Dims2::<p, UInt<uint, bit>>, Dims2::<UInt<uint, bit>, r>) => Dims2::<p, r>,
            #[generics(p: Dim, r: Dim, uint: Unsigned, bit: Bit)]
            (Dims2::<p, DynDim>, Dims2::<UInt<uint, bit>, r>) => Dims2::<p, r>,
            #[generics(p: Dim, r: Dim, uint: Unsigned, bit: Bit)]
            (Dims2::<p, UInt<uint, bit>>, Dims2::<DynDim, r>) => Dims2::<p, r>,
            #[generics(p: Dim, r: Dim)]
            (Dims2::<p, DynDim>, Dims2::<DynDim, r>) => Dims2::<p, r>,
        }
    }

    pub fn Flatten<input, start, end>(input: Dimensions, start: Dim, end: Dim) -> Dimensions {
        if IsDynDimensions(input) || IsDyn(start) || IsDyn(end) {
            DynDimensions
        } else {
            let input: DimsList = input;
            let start: Unsigned = start;
            let end: Unsigned = end;

            let heading: DimsList = list::Index(input, RangeTo::<start>);
            let trailing: DimsList = if end + 1u == list::Len(input) {
                Nil
            } else {
                let end_plus_1: Unsigned = end + 1u;
                list::Index(input, RangeFrom::<end_plus_1>)
            };
            let contracted: DimsList = list::Index(input, RangeInclusive::<(start, end)>);
            let product: Dim = list::ReduceProduct(contracted);
            list::Extend(heading, Cons::<product, trailing>)
        }
    }

    pub fn IndexSelect<input, dim, index>(input: Dimensions, dim: Dim, index: Dim) -> Dimensions {
        if IsDynDimensions(input) || IsDyn(dim) {
            DynDimensions
        } else {
            let input: DimsList = input;
            let dim: Unsigned = dim;
            let size: Dim = list::Get(input, dim);

            if IsDyn(index) || IsDyn(size) {
                list::Remove(input, dim)
            } else {
                match index < size {
                    B1 => list::Remove(input, dim)
                }
            }
        }
    }

    pub fn UnSqueeze<input, index>(input: Dimensions, index: Dim) -> Dimensions {
        if IsDynDimensions(input) || IsDyn(index) {
            DynDimensions
        } else {
            let input: DimsList = input;
            let index: Unsigned = index;
            if index == list::Len(input) {
                list::PushBack(input, 1u)
            } else {
                list::Insert(input, index, 1u)
            }
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
            DynDim => true,
            UTerm => false,
            #[generics(uint: Unsigned, bit: Bit)]
            UInt::<uint, bit> => false,
        }
    }
}

typ! {
    pub fn ContainsDyn<input>(input: DimsList) -> Bit {
        match input {
            #[generics(dim: Dim, tail: DimsList)]
            Cons::<dim, tail> => {
                if IsDyn(dim) {
                    true
                } else {
                    ContainsDyn(tail)
                }
            }
            Nil => false,
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
                        DynDim
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

                let lhs_rev: DimsList = list::Reverse(lhs);
                let rhs_rev: DimsList = list::Reverse(rhs);

                list::Reverse(PyTorchBroadcastRecursive(lhs_rev, rhs_rev))
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
                let dim: Dim = if IsDyn(ldim) || IsDyn(rdim) {
                    DynDim
                } else {
                    match (ldim, rdim) {
                        #[generics(uint: Unsigned, bit: Bit)]
                        (U1, UInt::<uint, bit>) => rdim,
                        #[generics(uint: Unsigned, bit1: Unsigned, bit2: Bit)]
                        (UInt::<UInt<uint, bit1>, bit2>, U1) => ldim,
                        #[generics(uint: Unsigned, bit1: Unsigned, bit2: Bit)]
                        (UInt::<UInt<uint, bit1>, bit2>, UInt::<UInt<uint, bit1>, bit2>) => ldim,
                    }
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
            DynDim
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
                let size: Dim = list::First(sizes);
                let padding: Dim = list::First(paddings);
                let dilation: Dim = list::First(dilations);
                let ksize: Dim = list::First(ksizes);
                let stride: Dim = list::First(strides);

                let new_sizes: DimsList = list::PopFront(sizes);
                let new_paddings: DimsList = list::PopFront(paddings);
                let new_dilations: DimsList = list::PopFront(dilations);
                let new_ksizes: DimsList = list::PopFront(ksizes);
                let new_strides: DimsList = list::PopFront(strides);

                let dim: Dim = ConvDim(size, padding, dilation, ksize, stride);
                let new_saved: DimsList = Cons::<dim, saved>;
                ConvDimsListRecursive(new_saved, new_sizes, new_paddings, new_dilations, new_ksizes, new_strides)
            }
            Nil => list::Reverse(saved),
        }
    }
}

typ! {
    pub fn Cat<inputs, index>(inputs: List, index: Dim) -> Dimensions {
        if AnyDynDimensions(inputs) || IsDyn(index) {
            DynDimensions
        } else {
            let index: Unsigned = index;

            // "transpose" the list of lists
            let zipped = list::ZipEx(inputs);

            // extract sublists of interests
            let leading_list: List = list::Index(zipped, RangeTo::<index>);
            let trailing_list: List = {
                let from: Unsigned = index + 1u;
                list::Index(zipped, RangeFrom::<from>)
            };
            let selected_dims: DimsList = list::Index(zipped, index);

            // merge dimensions
            let leading_dims = MapMergeDims(leading_list);
            let trailing_dims = MapMergeDims(trailing_list);
            let sum_dim: Dim = list::ReduceSum(selected_dims);

            // output
            list::Extend(list::PushBack(leading_dims, sum_dim), trailing_dims)
        }
    }

    fn MapMergeDims<inputs>(inputs: List) -> List {
        match inputs {
            #[generics(dims: DimsList, tail: List)]
            Cons::<dims, tail> => {
                let merged = MergeDims(dims);
                let new_tail = MapMergeDims(tail);
                Cons::<merged, new_tail>
            }
            Nil => Nil,
        }
    }

    fn MergeDims<dims>(dims: DimsList) -> Dim {
        let init: Dim = list::First(dims);
        MergeDimsRecursive(init, dims)
    }

    fn MergeDimsRecursive<curr, dims>(curr: Dim, dims: DimsList) -> Dim {
        match dims {
            #[generics(head: Dim, tail: DimsList)]
            Cons::<head, tail> => MergeDimsRecursive(MergeDim(curr, head), tail),
            Nil => curr,
        }
    }

    fn MergeDim<lhs, rhs>(lhs: Dim, rhs: Dim) -> Dim {
        if IsDyn(lhs) {
            if IsDyn(rhs) {
                DynDim
            } else {
                rhs
            }
        } else if IsDyn(rhs) {
            lhs
        } else {
            match lhs == rhs {
                B1 => lhs,
            }
        }
    }
}

typ! {
    pub fn CatAnyDyn<inputs>(inputs: List) -> List {
        let zipped = list::ZipEx(inputs);
        MapAnyDyn(zipped)
    }

    fn MapAnyDyn<inputs>(inputs: List) -> List {
        match inputs {
            #[generics(dims: DimsList, tail: List)]
            Cons::<dims, tail> => {
                let mapped = AnyDyn(dims);
                let new_tail = MapAnyDyn(tail);
                Cons::<mapped, new_tail>
            }
            Nil => Nil,
        }
    }
}

typ! {
    pub fn SqueezeAll<input>(input: Dimensions) -> Dimensions {
        if IsDynDimensions(input) {
            DynDimensions
        } else {
            let input: DimsList = input;
            if ContainsDyn(input) {
                DynDimensions
            } else {
                let input: StaticDimsList = input;
                SqueezeAllRecursive(Nil, input)
            }
        }
    }

    pub fn SqueezeAllRecursive<saved, remaining>(saved: StaticDimsList, remaining: StaticDimsList) -> StaticDimsList {
        match remaining {
            #[generics(dim: Unsigned, tail: StaticDimsList)]
            Cons::<dim, tail> => {
                let new_saved: StaticDimsList = if dim == 1u {
                    saved
                } else {
                    Cons::<dim, saved>
                };
                SqueezeAllRecursive(new_saved, tail)
            }
            Nil => list::Reverse(saved),
        }
    }
}

typ! {
    pub fn Squeeze<input, index>(input: Dimensions, index: Dim) -> Dimensions {
        if IsDynDimensions(input) || IsDyn(index) {
            DynDimensions
        } else {
            let input: DimsList = input;
            let index: Unsigned = index;
            SqueezeRecursive(Nil, input, index)
        }
    }

    pub fn SqueezeRecursive<saved, remaining, index>(saved: DimsList, remaining: DimsList, index: Unsigned) -> DimsList {
        if index == 0u {
            match remaining {
                #[generics(tail: DimsList)]
                Cons::<U1, tail> => list::Extend(list::Reverse(saved), tail),
                #[generics(tail: DimsList)]
                Cons::<DynDim, tail> => list::Extend(list::Reverse(saved), tail),
            }
        } else {
            match remaining {
                #[generics(dim: Dim, tail: DimsList)]
                Cons::<dim, tail> => {
                    let new_saved = Cons::<dim, saved>;
                    let new_index: Unsigned = index - 1u;
                    SqueezeRecursive(new_saved, tail, new_index)
                }
            }
        }
    }
}

typ! {
    pub fn AllDyn<dims>(dims: DimsList) -> Bit {
        match dims {
            #[generics(tail: DimsList)]
            Cons::<DynDim, tail> => AllDyn(tail),
            #[generics(tail: DimsList)]
            Cons::<UTerm, tail> => false,
            #[generics(uint: Unsigned, bit: Bit, tail: DimsList)]
            Cons::<UInt<uint, bit>, tail> => false,
            Nil => true,
        }
    }

    pub fn AnyDyn<dims>(dims: DimsList) -> Bit {
        match dims {
            #[generics(tail: DimsList)]
            Cons::<DynDim, tail> => true,
            #[generics(tail: DimsList)]
            Cons::<UTerm, tail> => AnyDyn(tail),
            #[generics(uint: Unsigned, bit: Bit, tail: DimsList)]
            Cons::<UInt<uint, bit>, tail> => AnyDyn(tail),
            Nil => false,
        }
    }

    pub fn AllDynDimensions<inputs>(inputs: List) -> Bit {
        match inputs {
            #[generics(tail: List)]
            Cons::<DynDimensions, tail> => AllDynDimensions(tail),
            #[generics(dim: Dim, dims_tail: DimsList, tail: List)]
            Cons::<Cons<dim, dims_tail>, tail> => false,
            Nil => true,
        }
    }

    pub fn AnyDynDimensions<inputs>(inputs: List) -> Bit {
        match inputs {
            #[generics(tail: List)]
            Cons::<DynDimensions, tail> => true,
            #[generics(dim: Dim, dims_tail: DimsList, tail: List)]
            Cons::<Cons<dim, dims_tail>, tail> => AnyDynDimensions(tail),
            Nil => false,
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
        let _: SameOp<
            CatAnyDynOp<List![Dims![1, 2, _, _], Dims![1, _, 3, _]]>,
            List![B0, B1, B1, B1],
        > = ();
        let _: SameOp<CatOp<List![Dims![1, 2, 3], Dims![?]], tyuint!(1)>, Dims![?]> = ();
        let _: SameOp<CatOp<List![Dims![1, 2, 3], Dims![1, 5, 3]], DynDim>, Dims![?]> = ();
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
        let _: SameOp<IndexSelectOp<Dims![?], U0, U2>, Dims![?]> = ();
        let _: SameOp<IndexSelectOp<Dims![2, 3, 4], DynDim, U2>, Dims![?]> = ();
        let _: SameOp<IndexSelectOp<Dims![3], U0, U2>, Dims![]> = ();
        let _: SameOp<IndexSelectOp<Dims![2, 3, 4], U1, U2>, Dims![2, 4]> = ();
        let _: SameOp<IndexSelectOp<Dims![2, _, 4], U1, U2>, Dims![2, 4]> = ();
        let _: SameOp<IndexSelectOp<Dims![2, 3, 4], U1, DynDim>, Dims![2, 4]> = ();
        let _: SameOp<IndexSelectOp<Dims![2, _, 4], U1, DynDim>, Dims![2, 4]> = ();
        let _: SameOp<UnSqueezeOp<Dims![?], U1>, Dims![?]> = ();
        let _: SameOp<UnSqueezeOp<Dims![3, 7, 4], DynDim>, Dims![?]> = ();
        let _: SameOp<UnSqueezeOp<Dims![3, 7, 4], U0>, Dims![1, 3, 7, 4]> = ();
        let _: SameOp<UnSqueezeOp<Dims![3, 7, 4], U2>, Dims![3, 7, 1, 4]> = ();
        let _: SameOp<UnSqueezeOp<Dims![3, 7, 4], U3>, Dims![3, 7, 4, 1]> = ();
        let _: SameOp<SqueezeAllOp<Dims![?]>, Dims![?]> = ();
        let _: SameOp<SqueezeAllOp<Dims![2, _, 3]>, Dims![?]> = ();
        let _: SameOp<SqueezeAllOp<Dims![1, 2, 1, 3, 1, 4, 1]>, Dims![2, 3, 4]> = ();
        let _: SameOp<SqueezeOp<Dims![?], U1>, Dims![?]> = ();
        let _: SameOp<SqueezeOp<Dims![2, 1, 3], DynDim>, Dims![?]> = ();
        let _: SameOp<SqueezeOp<Dims![2, 1, 3], U1>, Dims![2, 3]> = ();
        let _: SameOp<SqueezeOp<Dims![2, _, 3], U1>, Dims![2, 3]> = ();
        let _: SameOp<AllDynOp<Dims![]>, B1> = ();
        let _: SameOp<AllDynOp<Dims![_, 4, _]>, B0> = ();
        let _: SameOp<AllDynOp<Dims![_, _, _]>, B1> = ();
        let _: SameOp<AnyDynOp<Dims![]>, B0> = ();
        let _: SameOp<AnyDynOp<Dims![_, 4, _]>, B1> = ();
        let _: SameOp<AnyDynOp<Dims![_, _, _]>, B1> = ();
        let _: SameOp<AnyDynOp<Dims![3, 4, 5]>, B0> = ();
        let _: SameOp<AllDynDimensionsOp<List![Dims![?], Dims![_]]>, B0> = ();
        let _: SameOp<AllDynDimensionsOp<List![Dims![?], Dims![?]]>, B1> = ();
        let _: SameOp<AnyDynDimensionsOp<List![Dims![3], Dims![_]]>, B0> = ();
        let _: SameOp<AnyDynDimensionsOp<List![Dims![?], Dims![_]]>, B1> = ();
        let _: SameOp<AnyDynDimensionsOp<List![Dims![?], Dims![?]]>, B1> = ();
    }
}
