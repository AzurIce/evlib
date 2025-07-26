# DuckDB Integration with evlib

This document demonstrates how to use evlib's zero-copy Arrow integration with DuckDB for high-performance event data analysis.

## Overview

evlib provides seamless integration with the Apache Arrow ecosystem through zero-copy data transfer. This enables efficient interoperability with analytics engines like DuckDB, allowing you to perform complex SQL queries on event camera data without expensive data copies.

## Basic Usage

### Loading Events to Arrow Format

```python
import evlib

# Load events directly to Apache Arrow format
events = evlib.formats.load_events_to_arrow("/path/to/your/events.hdf5")
```

### DuckDB Integration

```python
import evlib
import duckdb

# Load events to Arrow format (zero-copy)
events = evlib.formats.load_events_to_arrow("/Users/tallam/github/tallamjr/origin/evlib/data/prophersee/samples/hdf5/pedestrians.hdf5")

# Basic queries
print("=== Basic Event Statistics ===")

# Total event count
total_events = duckdb.sql("SELECT COUNT(*) as total_events FROM events").fetchone()[0]
print(f"Total events: {total_events:,}")

# Event distribution by polarity
polarity_dist = duckdb.sql("""
    SELECT p as polarity, COUNT(*) as count,
           ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
    FROM events
    GROUP BY p
    ORDER BY p
""").fetchall()

print(f"\\nPolarity distribution:")
for pol, count, pct in polarity_dist:
    print(f"  Polarity {pol}: {count:,} events ({pct}%)")

# Temporal statistics
temporal_stats = duckdb.sql("""
    SELECT
        MIN(t) as min_timestamp,
        MAX(t) as max_timestamp,
        AVG(t) as avg_timestamp,
        STDDEV(t) as std_timestamp
    FROM events
""").fetchone()

print(f"\\nTemporal statistics:")
print(f"  Time range: {temporal_stats[0]:,} to {temporal_stats[1]:,}")
print(f"  Average timestamp: {temporal_stats[2]:,.0f}")
print(f"  Standard deviation: {temporal_stats[3]:,.0f}")

# Spatial statistics
spatial_stats = duckdb.sql("""
    SELECT
        MIN(x) as min_x, MAX(x) as max_x,
        MIN(y) as min_y, MAX(y) as max_y,
        COUNT(DISTINCT x) as unique_x_coords,
        COUNT(DISTINCT y) as unique_y_coords
    FROM events
""").fetchone()

print(f"\\nSpatial statistics:")
print(f"  X range: {spatial_stats[0]} to {spatial_stats[1]} ({spatial_stats[4]} unique values)")
print(f"  Y range: {spatial_stats[2]} to {spatial_stats[3]} ({spatial_stats[5]} unique values)")
```

## Advanced Analytics

### Time-based Analysis

```python
# Events per second
events_per_second = duckdb.sql("""
    SELECT
        t // 1000000 as second,  -- Convert microseconds to seconds
        COUNT(*) as events_count
    FROM events
    GROUP BY t // 1000000
    ORDER BY second
    LIMIT 10
""").fetchall()

print("Events per second (first 10 seconds):")
for second, count in events_per_second:
    print(f"  Second {second}: {count:,} events")
```

### Spatial Density Analysis

```python
# Spatial density heatmap data
density_map = duckdb.sql("""
    SELECT
        x // 10 * 10 as x_bin,  -- 10x10 pixel bins
        y // 10 * 10 as y_bin,
        COUNT(*) as event_density
    FROM events
    GROUP BY x_bin, y_bin
    HAVING event_density > 100  -- Filter low-density areas
    ORDER BY event_density DESC
    LIMIT 20
""").fetchall()

print("\\nTop 20 highest density regions (10x10 pixel bins):")
for x_bin, y_bin, density in density_map:
    print(f"  Region ({x_bin},{y_bin}): {density:,} events")
```

### Performance Comparison

The zero-copy Arrow integration provides significant performance benefits:

- **Memory efficiency**: No data copying between evlib and DuckDB
- **Query performance**: DuckDB's columnar engine optimized for Arrow data
- **Ecosystem compatibility**: Works seamlessly with Polars, Pandas, and other Arrow-compatible tools

### Alternative Ecosystems

The same Arrow data can be used with other analytics engines:

```python
# With Polars
import polars as pl
df = pl.from_arrow(events)
result = df.group_by("p").agg(pl.count())

# With Pandas (includes copy overhead)
import pandas as pd
df = events.to_pandas()
result = df.groupby("p").size()
```

## Supported File Formats

evlib's Arrow integration supports all major event camera formats:

- **HDF5**: Prophesee, iniVation datasets
- **EVT2/EVT3**: Prophesee RAW format
- **AEDAT**: iniVation format (v1.0, v2.0, v3.1, v4.0)
- **AER**: Address-Event Representation
- **Text**: Custom CSV-like formats

## Performance Notes

- Arrow integration is enabled with the `arrow` feature flag
- Zero-copy transfer requires compatible data layouts
- For maximum performance, use file formats that align with Arrow's columnar structure
- Large datasets benefit most from the zero-copy approach

## Dependencies

To use DuckDB integration:

```bash
pip install evlib[arrow] duckdb
```

Or for development:

```bash
maturin develop --features polars,arrow
pip install duckdb
```
