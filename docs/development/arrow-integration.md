# Apache Arrow Integration

## Overview

Apache Arrow integration in evlib provides zero-copy data transfer and high-performance interoperability with Arrow-based systems like PyArrow, DuckDB, and other Arrow ecosystem tools. This **complements** rather than **replaces** the existing efficient Polars infrastructure.

## When to Use Arrow vs Polars

### Current Polars Architecture (Optimal for Internal Processing)

```
File → Rust Format Reader → Rust Events Vec → Polars DataFrame → Python Dict → Polars LazyFrame
                                          ↑                      ↑
                                   Copy happens here      Copy happens here
                                   (~13 bytes/event)      (serialization)
```

**Use Polars for:**
- All `evlib.filtering` operations (most efficient)
- Internal evlib processing and analysis
- When working within the evlib ecosystem

### New Arrow Architecture (Optimal for External Integration)

```
File → Rust Format Reader → Rust Events Vec → Arrow RecordBatch → Python PyArrow Table
                                          ↑                     ↑
                                   Copy happens here       ZERO COPY!
                                   (~15 bytes/event)       (shared memory)
```

**Use Arrow for:**
- Exporting data to external Arrow-compatible systems
- Integration with DuckDB, Parquet files, or other Arrow tools
- When you need zero-copy data sharing with external libraries

## Python API

### Loading Events as Arrow

```python
import evlib.formats

# Load events directly as PyArrow Table (zero-copy from Rust)
arrow_table = evlib.formats.load_events_to_arrow("events.txt")

# With filtering options
arrow_table = evlib.formats.load_events_to_arrow(
    "events.h5",
    t_start=0.1,
    t_end=0.5,
    min_x=100, max_x=500,
    polarity=1
)
```

### Converting Arrow Back to Events

```python
# Convert PyArrow back to event data dictionary
events_dict = evlib.formats.pyarrow_to_events(arrow_table)

# This returns a dictionary with arrays:
# {"x": [100, 101, ...], "y": [200, 201, ...], "t": [0.001, 0.002, ...], "polarity": [1, 0, ...]}
```

### Integration with External Tools

```python
import evlib.formats
import duckdb
import pyarrow as pa

# Load as Arrow (zero-copy)
arrow_table = evlib.formats.load_events_to_arrow("events.h5")

# SQL queries with DuckDB (directly on Arrow data)
result = duckdb.sql("SELECT COUNT(*) FROM arrow_table WHERE polarity = 1")
print(f"Positive events: {result.fetchone()[0]}")

# Export to Parquet (very efficient)
arrow_table.write_parquet("events.parquet")

# Convert to Pandas if needed (not recommended for large datasets)
df = arrow_table.to_pandas()
```

## Schema Specification

### Arrow Schema Definition

The Arrow schema exactly matches the existing Polars schema for perfect compatibility:

```rust
Schema::new(vec![
    Field::new("x", DataType::Int16, false),
    Field::new("y", DataType::Int16, false),
    Field::new("timestamp", DataType::Duration(TimeUnit::Microsecond), false),
    Field::new("polarity", DataType::Int8, false),
])
```

### Memory Layout

| Field     | Type                         | Bytes | Description |
|-----------|------------------------------|-------|-------------|
| x         | Int16                        | 2     | Pixel x coordinate |
| y         | Int16                        | 2     | Pixel y coordinate |
| timestamp | Duration(Microseconds)       | 8     | Event timestamp |
| polarity  | Int8                         | 1     | Event polarity |
| **Total** |                              | **13** | Core data per event |
| **Arrow Overhead** |                   | **~2** | Array metadata |
| **Total with Arrow** |                 | **~15** | Bytes per event |

### Format-Specific Polarity Encoding

Arrow implementation maintains the same polarity encoding as the existing Polars implementation:

- **EVT2/EVT3/HDF5**: `0/1 → -1/1` (true polarity representation)
- **Text/Other**: `0/1 → 0/1` (matches file format)

### Timestamp Conversion

Automatic timestamp handling:
- **If timestamp ≥ 1,000,000**: Assume microseconds, use directly
- **If timestamp < 1,000,000**: Assume seconds, multiply by 1,000,000
- **Output**: Always Duration(Microseconds) for consistency

## Building with Arrow Support

### Installation

Arrow support requires the `arrow` feature flag:

```bash
# Install with Arrow support
maturin develop --features arrow,polars,python

# Or build for distribution
maturin build --release --features arrow,polars,python
```

### Feature Configuration

```toml
# Cargo.toml
[features]
default = ["polars", "python"]
arrow = ["dep:arrow", "dep:arrow-array", "dep:pyo3-arrow"]
zero-copy = ["arrow"]  # Alias for clarity

[dependencies]
arrow = { version = "55.0", default-features = false, optional = true }
arrow-array = { version = "55.0", optional = true }
pyo3-arrow = { version = "0.10.1", optional = true }
```

## Performance Characteristics

### Memory Efficiency Comparison

| Implementation | Bytes per Event | Notes |
|----------------|-----------------|-------|
| Raw Event struct | 24 | Rust representation |
| **Polars DataFrame** | **13** | **Most efficient for processing** |
| Arrow RecordBatch | 15 | Slightly more overhead than Polars |

### Zero-Copy Benefits

The real benefit of Arrow is **eliminating data duplication**:

- **Before**: Rust events → copy to Python lists → copy to NumPy → copy to external tool = **3x memory**
- **After**: Rust Arrow → zero-copy share with Python → direct external tool access = **1x memory**

### Performance Targets

- **Direct loading**: Match Polars performance (baseline)
- **Zero-copy scenarios**: 10-50% improvement vs dictionary conversion
- **Streaming**: Support datasets >100M events without memory issues

## Use Cases and Examples

### Scientific Computing with DuckDB

```python
import evlib.formats
import duckdb

# Load large event dataset
events = evlib.formats.load_events_to_arrow("large_dataset.h5")

# Complex SQL analysis (impossible with pure Python/Polars)
analysis = duckdb.sql("""
    SELECT
        x // 50 as tile_x,
        y // 50 as tile_y,
        COUNT(*) as event_count,
        AVG(EXTRACT(microseconds FROM timestamp)) as avg_timestamp
    FROM events
    WHERE polarity = 1
    GROUP BY tile_x, tile_y
    ORDER BY event_count DESC
    LIMIT 10
""").fetchdf()

print("Top 10 most active tiles:")
print(analysis)
```

### Data Export Pipeline

```python
import evlib.formats

# Load and export to multiple formats efficiently
events = evlib.formats.load_events_to_arrow("input.evt2")

# Export to Parquet (compressed, columnar storage)
events.write_parquet("output.parquet")

# Export to CSV (if needed)
events.to_pandas().to_csv("output.csv")

# Export to HDF5 via PyTables (if needed)
import tables
events.to_pandas().to_hdf("output.h5", key="events")
```

### Integration with Machine Learning

```python
import evlib.formats
import numpy as np
from sklearn.cluster import KMeans

# Load events as Arrow
events = evlib.formats.load_events_to_arrow("data.h5")

# Convert to NumPy for ML (zero-copy when possible)
coords = np.column_stack([
    events['x'].to_numpy(),
    events['y'].to_numpy()
])

# Spatial clustering
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(coords)

print(f"Found {len(np.unique(clusters))} spatial clusters")
```

## Architecture Details

### Core Components

1. **ArrowEventBuilder**: High-performance Arrow array construction
2. **ArrowEventStreamer**: Chunked processing for large datasets
3. **Schema Management**: Exact Polars compatibility
4. **Python Bindings**: pyo3-arrow integration for zero-copy transfer

### Integration Points

The Arrow implementation integrates at several levels:

```rust
// File loading with Arrow output
#[cfg(feature = "arrow")]
pub fn load_events_to_arrow(
    path: &str,
    config: &LoadConfig
) -> Result<RecordBatch, Box<dyn std::error::Error>>

// Python bindings for PyArrow
#[cfg(all(feature = "python", feature = "arrow"))]
#[pyfunction]
pub fn load_events_to_pyarrow(
    py: Python<'_>,
    path: &str,
    // ... filtering parameters
) -> PyResult<PyObject>  // Returns PyArrow Table
```

### Error Handling

Comprehensive error handling for Arrow operations:

```rust
#[derive(Debug, thiserror::Error)]
pub enum ArrowBuilderError {
    #[error("Arrow array construction failed: {0}")]
    ArrayConstruction(String),

    #[error("Invalid event data: {message}")]
    InvalidData { message: String },

    #[error("Feature not enabled: Arrow support requires 'arrow' feature flag")]
    FeatureNotEnabled,
}
```

## Testing and Validation

### Test Coverage

The Arrow integration includes comprehensive tests:

- **Schema validation**: Ensure exact match with Polars schema
- **Polarity encoding**: Verify format-specific encoding (EVT2 vs Text)
- **Timestamp conversion**: Test seconds ↔ microseconds conversion
- **Round-trip conversion**: Arrow → Events → Arrow consistency
- **Streaming**: Large dataset processing validation
- **Zero-copy verification**: Memory usage validation

### Running Tests

```bash
# Test Arrow functionality (without Python dependencies)
cargo test test_arrow_integration --no-default-features --features arrow

# Test specific Arrow features
cargo test test_arrow_schema_creation --no-default-features --features arrow
cargo test test_arrow_round_trip_conversion --no-default-features --features arrow
```

## Best Practices

### When to Use Arrow

✅ **Use Arrow when:**
- Exporting data to external systems (DuckDB, Parquet, etc.)
- Integrating with non-Polars data science tools
- Need zero-copy data sharing
- Working with Arrow-native applications

❌ **Don't use Arrow when:**
- Using `evlib.filtering` operations (Polars is more efficient)
- Working purely within the evlib ecosystem
- Memory usage is more critical than interoperability

### Performance Tips

1. **Prefer Polars for internal processing**: The existing Polars pipeline is optimized for evlib operations
2. **Use Arrow for export/import**: Convert to Arrow only when interfacing with external systems
3. **Consider streaming**: For large datasets (>5M events), streaming provides memory efficiency
4. **Batch operations**: Process multiple files together when possible

### Migration Guidelines

The Arrow integration is **additive** - no existing code needs to change:

```python
# Existing code continues to work (recommended for internal processing)
import evlib
events = evlib.load_events("data.h5")  # Returns Polars LazyFrame

# New Arrow API for external integration
import evlib.formats
arrow_events = evlib.formats.load_events_to_arrow("data.h5")  # Returns PyArrow Table
```

## Conclusion

The Apache Arrow integration provides powerful interoperability capabilities while maintaining the efficiency of the existing Polars-based architecture. Use Arrow when you need to interface with external systems, and continue using the Polars pipeline for internal evlib operations.

The zero-copy benefits are realized specifically when sharing data with external Arrow-compatible systems, enabling efficient data export, SQL analysis, and integration with the broader data science ecosystem.
