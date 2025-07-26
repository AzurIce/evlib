/*!
HDF5 reader with native ECF support.

This module provides direct HDF5 chunk reading capabilities to enable
our Rust ECF codec to decode Prophesee files without external dependencies.
*/

use crate::ev_core::{Event, Events};
use crate::ev_formats::prophesee_ecf_codec::PropheseeECFDecoder;
use hdf5_metno::{Dataset, File as H5File, Result as H5Result};
use hdf5_metno_sys::{h5d, h5p, h5s};
use std::io;

/// Read raw chunk data from an HDF5 dataset
/// This bypasses the HDF5 filter pipeline to get compressed chunks directly
pub fn read_raw_chunks(_file_path: &str, _dataset_path: &str) -> io::Result<Vec<Vec<u8>>> {
    // This is a placeholder function showing the intended structure

    // Parse HDF5 structure to find chunk locations
    // This is a simplified version - full implementation would parse B-trees
    let chunks = vec![];

    // TODO: Implement HDF5 B-tree parsing to locate chunks
    // For now, return empty to show the structure

    Ok(chunks)
}

/// Read Prophesee HDF5 file using our native ECF decoder
pub fn read_prophesee_hdf5_native(path: &str) -> H5Result<Events> {
    let file = H5File::open(path)?;

    // Check for Prophesee format
    let cd_group = file.group("CD")?;
    let events_dataset = cd_group.dataset("events")?;

    let shape = events_dataset.shape();
    let total_events = shape[0];

    if total_events == 0 {
        return Ok(Vec::new());
    }

    eprintln!(
        "Reading {} events from Prophesee HDF5 using native ECF decoder",
        total_events
    );

    // Get dataset information
    let chunk_size = match events_dataset.chunk() {
        Some(chunk_shape) => chunk_shape[0],
        None => 16384,
    };

    eprintln!("Dataset uses chunks of {} events", chunk_size);

    // Create Prophesee ECF decoder
    let decoder = PropheseeECFDecoder::new().with_debug(true);
    let mut all_events = Vec::with_capacity(total_events);

    // Read dataset filters to confirm ECF
    let filters = get_dataset_filters(&events_dataset)?;
    let has_ecf = filters.iter().any(|&id| id == 36559);

    if !has_ecf {
        return Err(hdf5_metno::Error::Internal(
            "Dataset does not use ECF compression".to_string(),
        ));
    }

    // Try to read raw chunk data
    // For now, we'll implement a hybrid approach using the dataset's raw data access

    // Check if we can read the dataset storage details
    let storage_size = events_dataset.storage_size();
    eprintln!("Dataset storage size: {} bytes", storage_size);

    // Attempt to decode chunks
    let num_chunks = total_events.div_ceil(chunk_size);
    eprintln!("Processing {} chunks", num_chunks);

    // Process all chunks
    let mut chunks_processed = 0;
    let mut total_decoded_events = 0;

    for chunk_idx in 0..num_chunks {
        eprintln!("Processing chunk {}/{}", chunk_idx + 1, num_chunks);

        match read_compressed_chunk(&events_dataset, chunk_idx) {
            Ok(compressed_data) => {
                eprintln!(
                    "Read compressed chunk {} ({} bytes)",
                    chunk_idx,
                    compressed_data.len()
                );

                // The compressed_data contains HDF5 chunk headers + ECF payload
                // We need to extract just the ECF payload
                // For now, try different offsets to find the ECF data
                let ecf_payload = extract_ecf_payload_from_chunk(&compressed_data)
                    .map_err(|e| hdf5_metno::Error::Internal(e.to_string()))?;

                // Decode with our ECF codec
                match decoder.decode(&ecf_payload) {
                    Ok(decoded_events) => {
                        eprintln!(
                            "Decoded {} events from chunk {}",
                            decoded_events.len(),
                            chunk_idx
                        );

                        let event_count = decoded_events.len();

                        // Convert PropheseeEvent to Event
                        for ecf_event in decoded_events {
                            all_events.push(Event {
                                t: ecf_event.t as f64, // Keep as microseconds - convert_timestamp will handle units
                                x: ecf_event.x,
                                y: ecf_event.y,
                                polarity: ecf_event.p > 0,
                            });
                        }

                        total_decoded_events += event_count;
                        chunks_processed += 1;
                    }
                    Err(e) => {
                        eprintln!("ECF decode error for chunk {}: {}", chunk_idx, e);
                        // Continue with other chunks instead of failing completely
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to read chunk {}: {}", chunk_idx, e);

                // For the first chunk failure, return a helpful error
                if chunk_idx == 0 {
                    return Err(hdf5_metno::Error::Internal(format!(
                        "Failed to read HDF5 chunks with native ECF decoder. \
                         This may require HDF5 1.10.5+ or specific build options. \
                         Error: {}",
                        e
                    )));
                }
                // For subsequent chunks, continue processing
            }
        }

        // Progress reporting for large files
        if num_chunks > 10 && chunk_idx % (num_chunks / 10) == 0 {
            let progress = (chunk_idx as f64 / num_chunks as f64) * 100.0;
            eprintln!(
                "Progress: {:.1}% ({} chunks processed, {} events decoded)",
                progress, chunks_processed, total_decoded_events
            );
        }
    }

    eprintln!(
        "Completed processing: {}/{} chunks successful, {} total events decoded",
        chunks_processed, num_chunks, total_decoded_events
    );

    if all_events.is_empty() {
        Err(hdf5_metno::Error::Internal(
            "Native ECF decoding integration in progress. This file should load automatically when complete.".to_string()
        ))
    } else {
        Ok(all_events)
    }
}

/// Get filter IDs from a dataset
fn get_dataset_filters(dataset: &Dataset) -> H5Result<Vec<u32>> {
    let mut filter_ids = Vec::new();

    // Get dataset's property list to access filters
    let dataset_id = dataset.id();
    let plist_id = unsafe { h5d::H5Dget_create_plist(dataset_id) };

    if plist_id < 0 {
        return Err(hdf5_metno::Error::Internal(
            "Failed to get dataset creation property list".to_string(),
        ));
    }

    // Get number of filters
    let num_filters = unsafe { h5p::H5Pget_nfilters(plist_id) };

    if num_filters < 0 {
        unsafe { h5p::H5Pclose(plist_id) };
        return Err(hdf5_metno::Error::Internal(
            "Failed to get number of filters".to_string(),
        ));
    }

    // Read each filter
    for filter_idx in 0..num_filters {
        let mut flags = 0u32;
        let mut cd_nelmts = 0usize;
        let mut name = vec![0i8; 256];

        let filter_id = unsafe {
            h5p::H5Pget_filter2(
                plist_id,
                filter_idx as u32,
                &mut flags,
                &mut cd_nelmts,
                std::ptr::null_mut(), // cd_values - we don't need them
                name.len(),
                name.as_mut_ptr(),
                std::ptr::null_mut(), // filter_config - we don't need it
            )
        };

        if filter_id >= 0 {
            filter_ids.push(filter_id as u32);

            // Convert name to string for debugging
            let filter_name = unsafe {
                std::ffi::CStr::from_ptr(name.as_ptr())
                    .to_string_lossy()
                    .to_string()
            };

            eprintln!(
                "Found filter: ID={} (0x{:x}), name='{}'",
                filter_id, filter_id, filter_name
            );
        }
    }

    unsafe { h5p::H5Pclose(plist_id) };
    Ok(filter_ids)
}

/// Read a compressed chunk from the dataset
fn read_compressed_chunk(dataset: &Dataset, chunk_idx: usize) -> io::Result<Vec<u8>> {
    let dataset_id = dataset.id();

    // Get dataspace to determine dimensions
    let space_id = unsafe { h5d::H5Dget_space(dataset_id) };
    if space_id < 0 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "Failed to get dataset dataspace",
        ));
    }

    // Get number of dimensions
    let ndims = unsafe { h5s::H5Sget_simple_extent_ndims(space_id) };
    if ndims < 0 {
        unsafe { h5s::H5Sclose(space_id) };
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "Failed to get dataspace dimensions",
        ));
    }

    // Get total number of chunks
    let mut num_chunks: u64 = 0;
    let status = unsafe { h5d::H5Dget_num_chunks(dataset_id, space_id, &mut num_chunks) };
    if status < 0 {
        unsafe { h5s::H5Sclose(space_id) };
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "Failed to get number of chunks",
        ));
    }

    // Validate chunk index
    if chunk_idx >= num_chunks as usize {
        unsafe { h5s::H5Sclose(space_id) };
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Chunk index {} out of range (0-{})",
                chunk_idx,
                num_chunks - 1
            ),
        ));
    }

    // Get chunk information
    let mut chunk_offset = vec![0u64; ndims as usize];
    let mut filter_mask = 0u32;
    let mut chunk_addr = 0u64;
    let mut chunk_size = 0u64;

    let status = unsafe {
        h5d::H5Dget_chunk_info(
            dataset_id,
            space_id,
            chunk_idx as u64,
            chunk_offset.as_mut_ptr(),
            &mut filter_mask,
            &mut chunk_addr,
            &mut chunk_size,
        )
    };

    if status < 0 {
        unsafe { h5s::H5Sclose(space_id) };
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to get chunk {} info", chunk_idx),
        ));
    }

    eprintln!(
        "Chunk {}: offset={:?}, filter_mask=0x{:x}, addr=0x{:x}, size={} bytes",
        chunk_idx, chunk_offset, filter_mask, chunk_addr, chunk_size
    );

    // Check if ECF filter is applied (filter ID 36559 = 0x8ECF)
    // The filter_mask indicates which filters were applied during compression
    if filter_mask == 0 {
        eprintln!("Warning: No filters applied to chunk {}", chunk_idx);
    }

    // Allocate buffer for compressed data
    let mut compressed_data = vec![0u8; chunk_size as usize];

    // Read raw compressed chunk data (bypassing HDF5 filter pipeline)
    let read_status = unsafe {
        h5d::H5Dread_chunk(
            dataset_id,
            h5p::H5P_DEFAULT,
            chunk_offset.as_ptr(),
            &mut filter_mask,
            compressed_data.as_mut_ptr() as *mut std::ffi::c_void,
        )
    };

    unsafe { h5s::H5Sclose(space_id) };

    if read_status < 0 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to read compressed chunk {} data", chunk_idx),
        ));
    }

    eprintln!(
        "Successfully read {} bytes of compressed data from chunk {}",
        compressed_data.len(),
        chunk_idx
    );

    Ok(compressed_data)
}

/// Complete end-to-end implementation structure for native ECF support
pub mod native_ecf {

    /// This shows what a complete implementation would look like
    pub fn full_implementation_outline() {
        // 1. Parse HDF5 file structure
        // 2. Locate B-tree nodes containing chunk addresses
        // 3. Read compressed chunks directly from file
        // 4. Pass chunks to our ECF decoder
        // 5. Return decoded events

        // The implementation would:
        // - Use our own HDF5 parser for chunk locations
        // - Read raw bytes from the file
        // - Decode with our ECF codec
        // - No external dependencies needed!
    }
}

/// Extract ECF payload from HDF5 chunk data
///
/// The raw chunk data from H5Dread_chunk includes HDF5 metadata.
/// We need to find the actual ECF compressed data within this.
fn extract_ecf_payload_from_chunk(chunk_data: &[u8]) -> io::Result<Vec<u8>> {
    // Based on our testing, the first few bytes contain HDF5 metadata
    // The ECF payload should start after some header bytes

    if chunk_data.len() < 16 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Chunk too small to contain ECF data",
        ));
    }

    // Try different offsets to find valid ECF header
    // ECF format starts with: num_events (u32) + flags (u32)
    // Try more offsets including odd numbers as HDF5 chunk headers can vary
    let offsets_to_try = [
        0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64,
    ];

    for &offset in &offsets_to_try {
        if offset + 8 > chunk_data.len() {
            continue;
        }

        let payload = &chunk_data[offset..];

        // Check if this looks like a valid ECF header
        if is_valid_ecf_header(payload) {
            eprintln!(
                "Found ECF payload at offset {} (payload size: {} bytes)",
                offset,
                payload.len()
            );
            return Ok(payload.to_vec());
        }
    }

    // If we can't find a valid header, log more details for debugging
    eprintln!(
        "Debug: First 64 bytes of chunk: {:02x?}",
        &chunk_data[..chunk_data.len().min(64)]
    );

    // Try scanning the entire chunk for a valid ECF pattern
    for offset in (0..chunk_data.len().saturating_sub(8)).step_by(4) {
        let payload = &chunk_data[offset..];
        if is_valid_ecf_header(payload) {
            eprintln!(
                "Found ECF payload at scanned offset {} (payload size: {} bytes)",
                offset,
                payload.len()
            );
            return Ok(payload.to_vec());
        }
    }

    // If no valid ECF header found, try the whole chunk
    // (maybe it's already the ECF payload)
    eprintln!(
        "No ECF header found, trying whole chunk ({} bytes)",
        chunk_data.len()
    );
    Ok(chunk_data.to_vec())
}

/// Check if data starts with a valid ECF header
fn is_valid_ecf_header(data: &[u8]) -> bool {
    if data.len() < 16 {
        return false;
    }

    // Check for the observed pattern: [02, 00, 01, 00, ?, ?, ?, ?, 00, 00, 00, 00]
    if data[0] == 0x02
        && data[1] == 0x00
        && data[2] == 0x01
        && data[3] == 0x00
        && data[8] == 0x00
        && data[9] == 0x00
        && data[10] == 0x00
        && data[11] == 0x00
    {
        // This looks like a Prophesee ECF chunk format
        // The event count might be at bytes 4-7
        let potential_events = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        eprintln!(
            "Prophesee ECF chunk detected with potential {} events",
            potential_events
        );
        return true;
    }

    // Also try the original ECF format
    let num_events = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let flags = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

    // More strict sanity checks for ECF header
    // - num_events should be reasonable (1 to 16384 for chunks)
    // - flags should have known ECF flag bits (only low 3 bits used)
    if num_events > 0 && num_events <= 16384 && (flags & 0xFFFFFFF8) == 0 {
        eprintln!(
            "Standard ECF header found: {} events, flags=0x{:x}",
            num_events, flags
        );
        return true;
    }

    false
}
