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
macro_rules! dims {
    [] => {
        type_freak::list::Nil
    };
    [$dim:literal $(, $remaining:tt)* $(,)?] => {
        type_freak::list::Cons { head: tyuint!($dim), tail: dims![$($remaining),*] }
    };
}
