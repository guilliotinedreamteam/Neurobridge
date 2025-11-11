# NeuroBridge API Documentation

## Overview

The NeuroBridge API provides endpoints for neural signal processing and data analysis.

## Base URL

```
http://localhost:5000/api
```

## Authentication

Currently, the API does not require authentication for local development. Production deployments should implement proper authentication.

## Endpoints

### Health Check

**GET** `/health`

Returns the health status of the API.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Process Signal

**POST** `/process`

Process neural signal data.

**Request Body:**
```json
{
  "signal": [0.1, 0.2, 0.3, ...],
  "sampling_rate": 1000,
  "filters": ["bandpass", "notch"]
}
```

**Response:**
```json
{
  "processed_signal": [0.15, 0.25, 0.35, ...],
  "metadata": {
    "duration": 1.5,
    "samples": 1500
  }
}
```

### Analyze Data

**POST** `/analyze`

Perform analysis on neural data.

**Request Body:**
```json
{
  "data": [...],
  "analysis_type": "frequency_domain"
}
```

**Response:**
```json
{
  "results": {...},
  "visualization_data": {...}
}
```

## Error Handling

All endpoints return standard HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid input
- `500 Internal Server Error`: Server error

Error response format:
```json
{
  "error": "Error message",
  "code": "ERROR_CODE"
}
```

## Rate Limiting

Currently no rate limiting is implemented for local development.

## WebSocket API

For real-time data streaming, connect to:

```
ws://localhost:5000/stream
```

### Events

- `connect`: Connection established
- `data`: New data available
- `error`: Error occurred
- `disconnect`: Connection closed

## Examples

See the [Examples](./EXAMPLES.md) documentation for code samples.