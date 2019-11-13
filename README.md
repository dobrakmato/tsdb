

### Data structure

Data points are organized into individual **Series**. These series are independent 
of each other and can have different data-values with different timestamps. Each
**Series** has a name and data schema associated with it.

#### Schemas

Currently these schemas are supported: `f32`.


### Operations on the data

Database provides following operations on the data.

#### Create series

Creates a new **Series** object in the database.

```
CREATE SERIES {series} {schema}
CREATE SERIES outdoor_sunlight f32
```

#### List series

Returns a list of all available series inside the database.

```
SHOW SERIES
```

#### Insert value

Inserts a specified value to the specified time **Series** object.

```
INSERT INTO {series} {value}
INSERT INTO outdoor_sunlight 4806.0
```

#### Query values

Gets all data points inside the specified **Series** object in specified range of time.

```
SELECT {series} BETWEEN {start_timestamp} AND {end_timestamp}
SELECT outdoor_sunlight BETWEEN 01-01-2019 AND 02-01-2019
```

It is also possible to query result of running an aggregate function on specified range data points in the specified **Series** object.

```
SELECT [AVG/SUM/MAX/MIN/STDDEV] {series} BETWEEN {start_timestamp} AND {end_timestamp} FOR EACH [MINUTE/HOUR/DAY/WEEK/MONTH]
SELECT AVG outdoor_sunlight BETWEEN 01-01-2019 AND 02-01-2019 FOR EACH HOUR
```