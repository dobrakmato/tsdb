time-series database
---------------------

Features:
- [ ] ACID compliant
- [ ] high performance
- [ ] multi-threaded
- [ ] low memory usage
- [ ] query cache
- [ ] textual command interface
- [ ] binary command interface


### ACID

#### Durability

Server employs redo-log to allow for disaster recovery. Commands are not confirmed
(and thus should be treated as not executed by clients) until they are written to
redo log and flushed to non-volatile memory.

### Data structure

Data points are organized into individual **Series**. These series are independent 
of each other and can have different data-values with different timestamps. Each
**Series** has a name and data schema associated with it.

Data points in one **Series** object is split into multiple **Blocks**. Each block
has a size of `4096` bytes. Multiple blocks (`8192`) are stored in one file. The
data points can be stored inside multiple files. Some blocks can be loaded in memory
and others can be stored only on disk. 

Timestamp data is delta-encoded using varints (integers with variable size). Encoding 
of actual data-points is schema dependant.

The database also stores **Index** which is used to speed up queries on the data. Index
object stores lowest and highest timestamp for each block. There is one **Index** for each
**Series** object. Also the index is stored in one file and it is loaded in memory at all
times.

#### Schemas

Currently these schemas are supported: `f32`.

##### f32

- Alignment: `4 bytes`
- Storage: 1024 values per block 
- Encoding: low-endian without any compression


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