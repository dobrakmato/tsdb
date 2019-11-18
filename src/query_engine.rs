use crate::aggregates::{Average, Min, Max, Count, Sum, Aggregate};

/// All supported aggregate functions by the database.
pub enum AggFn {
    Avg(Average),
    Min(Min),
    Max(Max),
    Count(Count),
    Sum(Sum),
}

impl Aggregate<f32> for AggFn {
    fn next(&mut self, item: f32) {
        match self {
            AggFn::Avg(x) => x.next(item),
            AggFn::Min(x) => x.next(item),
            AggFn::Max(x) => x.next(item),
            AggFn::Count(x) => x.next(item),
            AggFn::Sum(x) => x.next(item),
        }
    }

    fn result(&mut self) -> f32 {
        match self {
            AggFn::Avg(x) => x.result(),
            AggFn::Min(x) => x.result(),
            AggFn::Max(x) => x.result(),
            AggFn::Count(x) => x.result(),
            AggFn::Sum(x) => x.result(),
        }
    }
}

/// All kinds that are selectable and thus can be part of select result.
pub enum Selectable {
    Value,
    Timestamp,
    Aggregate(AggFn),
}

/// All possible group by intervals.
pub enum GroupBy {
    MINUTE,
    HOUR,
    DAY,
    WEEK,
    MONTH,
}

/// Struct representing FROM <min>, UNTIL <max>, BETWEEN <min> AND <max> clauses.
pub struct Between {
    pub min_timestamp: Option<usize>,
    pub max_timestamp: Option<usize>,
}

/// Struct that represents a select query.
pub struct Select<'a> {
    pub select: Vec<Selectable>,
    pub from: &'a str,
    pub between: Option<Between>,
    pub group_by: Option<GroupBy>,
}