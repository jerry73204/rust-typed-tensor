#[macro_export]
macro_rules! DimsT {
    [] => {
        type_freak::list::Nil
    };
    [?] => {
        $crate::dim::DynDimensions
    };
    [$dim:ty $(, $remaining:ty)* $(,)?] => {
        type_freak::list::Cons<$dim, DimsT![$($remaining),*]>
    };
}

#[macro_export]
macro_rules! Dims {
    [] => {
        type_freak::list::Nil
    };
    [?] => {
        $crate::dim::DynDimensions
    };
    [$dim:literal $(, $remaining:tt)* $(,)?] => {
        type_freak::list::Cons<tyuint!($dim), Dims![$($remaining),*]>
    };
    [_ $(, $remaining:tt)* $(,)?] => {
        type_freak::list::Cons<$crate::dim::Dyn, Dims![$($remaining),*]>
    };
}

// TODO: dynamic dim
#[macro_export]
macro_rules! dims_t {
    [] => {
        type_freak::list::Nil
    };
    [$dim:expr $(, $remaining:expr)* $(,)?] => {
        type_freak::list::Cons { head: $dim, tail: dims_t![$($remaining),*] }
    };
}

// TODO: dynamic dim
#[macro_export]
macro_rules! dims {
    [] => {
        type_freak::list::Nil
    };
    [$dim:literal $(, $remaining:tt)* $(,)?] => {
        type_freak::list::Cons { head: tyuint!($dim), tail: dims![$($remaining),*] }
    };
}
