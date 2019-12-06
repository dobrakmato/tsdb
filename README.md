time-series database
---------------------

Features:
- [ ] ACID compliant
- [ ] high performance
- [ ] multi-threaded
- [ ] query cache
- [ ] B-Tree index
- [x] sorted array index (bin-search)
- [x] one scalar per data-point
- [ ] aggregate functions
- [ ] `WHERE` on timestamp
- [ ] `WHERE` on value
- [ ] `GROUP BY` support on timestamp
- [x] textual command interface
- [ ] binary command interface
- [ ] multiple scalars (tuples) per data-point
- [ ] old data downsampling
- [ ] old data deletion
- [ ] auth: name & password
- [ ] ssl / tls

### ACID

### Data structure

Data points are organized into individual **Series**. These series are independent 
of each other and can have different data-values with different timestamps. Each
**Series** has a name and data schema associated with it.

Schema of the **Series** dictates how are the data points encoded on the disk.

Data points in one **Series** are split into multiple **Blocks**. Each block
has a size of `4096` bytes. These block form a *log-structured merge-tree*. 
Multiple blocks (`2048`) are stored in one file. During the application execution
some blocks are loaded in memory (cached). All blocks are stored on disk.

The database also stores **Index** which is used to speed up queries on the data. Index
object stores lowest and highest timestamp for each block. There is one **Index** for each
**Series** object. Also the index is stored in one file and it is loaded in memory at all
times.

#### Schemas

Currently these schemas are supported: `f32`.

Planned schemas: `f32`, `i32`, `f64`, `i64`, `bool`.

##### f32

- Timestamp encoding: delta-encoded LEB128 varints
- Value encoding: low-endian without any compression

##### i32

- Timestamp encoding: delta-encoded LEB128 varints
- Value encoding: delta-encoded LEB128 varints

##### bool

- Timestamp encoding: delta-encoded LEB128 varints
- Value encoding: One boolean takes one bit.


### Operations on the data

Database provides following operations on the data.

#### Create series

Creates a new **Series** object in the database.

```
CREATE SERIES {series} {schema}
CREATE SERIES outdoor_sunlight f32
```

```json
{
  "CreateSeries": "default"
}
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

```json
{
  "Insert": {
    "to": "default",
    "value": 3.14
  }
}
```


#### Query values

Gets all data points inside the specified **Series** object in specified range of time.

```
SELECT {series} BETWEEN {start_timestamp} AND {end_timestamp}
SELECT outdoor_sunlight BETWEEN 01-01-2019 AND 02-01-2019
```

```json
{
  "Select": {
    "from": "default",
    "between": {
      "min": null,
      "max": null
    }
  }
}
```

It is also possible to query result of running an aggregate function on specified range data points in the specified **Series** object.

```
SELECT [AVG/SUM/MAX/MIN/COUNT] {series} BETWEEN {start_timestamp} AND {end_timestamp} GROUP BY [MINUTE/HOUR/DAY/WEEK/MONTH]
SELECT AVG outdoor_sunlight BETWEEN 01-01-2019 AND 02-01-2019 GROUP BY HOUR
```