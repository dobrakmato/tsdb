use crate::engine::{Timestamp, Schema, Point};
use crate::engine::server::{Engine, SimpleServer};
use std::fmt::Display;
use static_assertions::_core::fmt::{Formatter, Error};
use std::ops::{AddAssign, Sub, Div};
use std::borrow::BorrowMut;
use crate::query_engine::aggregates::{Count, Average, Aggregate};

/// Contains all aggregate functions with their
/// structs (data types) that contain their internal state.
///
/// All aggregate functions implement one trait called "Aggregate"
/// which ensure that all of them have the same interface.
pub mod aggregates {
    use std::ops::{AddAssign, Sub, Div};

    pub trait One {
        const ONE: Self;
    }

    pub trait Zero {
        const ZERO: Self;
    }

    macro_rules! impl_one_zero {
    ($typ:ty) => {
        impl One for $typ {
            const ONE: $typ = 1 as $typ;
        }

        impl Zero for $typ {
            const ZERO: $typ = 0 as $typ;
        }
    };
    ($typ:ty, $($y:ty),+) => (
        impl_one_zero!($typ);
        impl_one_zero!($($y),+);
    )
}

    impl_one_zero!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64);

    /// Basic trait used for all aggregate functions.
    pub trait Aggregate<T> {
        type Output;
        fn item(&mut self, item: T);
        fn result(&mut self) -> Self::Output;
    }

    /// Aggregate function that computes arithmetic mean of values.
    pub struct Average<A, C = u64> {
        avg: A,
        count: C,
    }

    macro_rules! impl_average {
    ($typ:ty) => {
        impl<A, C> Aggregate<A> for $typ
            where A: Copy + AddAssign<A> + Sub<A, Output=A> + Div<C, Output=A>,
                  C: One + Copy + AddAssign<C>
        {
            type Output = A;

            fn item(&mut self, item: A) {
                self.avg += (item - self.avg) / self.count;
                self.count += C::ONE;
            }

            fn result(&mut self) -> Self::Output {
                self.avg
            }
        }
    };
}

    impl_average!(Average<A, C>);
    impl_average!(&mut Average<A, C>);


    impl<A, C> Default for Average<A, C> where A: Zero, C: One {
        fn default() -> Self {
            Average { avg: A::ZERO, count: C::ONE }
        }
    }

    /// Aggregate function that counts number of items.
    #[derive(Default)]
    pub struct Count<C = u64>(C);

    impl<C, T> Aggregate<T> for &mut Count<C>
        where C: One + AddAssign<C> + Copy
    {
        type Output = C;

        fn item(&mut self, _: T) {
            self.0 += C::ONE;
        }

        fn result(&mut self) -> Self::Output {
            self.0
        }
    }
}

pub enum AggFn<V> {
    Count(Count),
    Average(Average<V>),
}

impl<V> AggFn<V> where V: Copy + AddAssign<V> + Sub<V, Output=V> + Div<u64, Output=V> {
    fn next(&mut self, val: V) {
        match self {
            AggFn::Count(a) => a.borrow_mut().item(val),
            AggFn::Average(a) => a.borrow_mut().item(val),
        }
    }
}


impl<V> Display for AggFn<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        match self {
            AggFn::Count(_) => f.write_str("Count"),
            AggFn::Average(_) => f.write_str("Average"),
        }
    }
}

pub enum Selectable<V> {
    Timestamp,
    Value,
    AggFn(AggFn<V>),
}

pub enum From<V> {
    Series(String),
    Subquery(Box<Query<V>>),
}

pub enum GroupBy {
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Year,
}

#[derive(Default)]
pub struct Between {
    min_timestamp: Option<Timestamp>,
    max_timestamp: Option<Timestamp>,
}

pub struct Query<V> {
    from: From<V>,
    select: Vec<Selectable<V>>,
    between: Between,
    group_by: Option<GroupBy>,
}

struct Col<'a, A> { name: &'a str, data: Vec<A> }

pub trait Column<V> {
    fn push(&mut self, val: &Point<V>);
}

impl<'a, V> Column<V> for Col<'a, V> where V: Copy {
    fn push(&mut self, val: &Point<V>) {
        self.data.push(val.value);
    }
}

pub type QueryResult<V> = Vec<Box<dyn Column<V>>>;

pub trait QueryEngine<V> {
    fn handle_query(&mut self, query: Query<V>) -> QueryResult<V>;
}

impl<S, V> Engine<S, V> where S: Schema<V> {}

impl<S, V: 'static> QueryEngine<V> for Engine<S, V>
    where S: Schema<V>,
          V: Copy + AddAssign<V> + Sub<V, Output=V> + Div<u64, Output=V>
{
    fn handle_query(&mut self, query: Query<V>) -> QueryResult<V> {
        let mut res = QueryResult::new();

        // prepare result vectors / agg fns
        query.select.iter().for_each(|it| {
            match it {
                Selectable::Timestamp => res.push(Box::new(Col::<V> { name: "timestamp", data: vec![] })),
                Selectable::Value => res.push(Box::new(Col::<V> { name: "value", data: vec![] })),
                _ => {}
            }
        });

        if let From::Series(t) = query.from {
            let mut selectables = query.select;
            self.retrieve_points(&t,
                                 query.between.min_timestamp,
                                 query.between.max_timestamp)
                .iter()
                .for_each(|p| {
                    selectables.iter_mut()
                        .enumerate()
                        .for_each(|(idx, s)| {
                            match s {
                                Selectable::Timestamp | Selectable::Value => res[idx].push(p),
                                Selectable::AggFn(t) => t.next(p.value),
                            }
                        });
                });
        }

        res
    }
}
