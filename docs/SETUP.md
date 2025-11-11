# NeuroBridge Setup Guide

This guide will help you set up the NeuroBridge project on your local machine.

## Prerequisites

- Python 3.9 or higher
- Node.js 18 or higher
- npm or yarn package manager
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/neurobridge.git
cd neurobridge
```

### 2. Backend Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Or using a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Frontend Setup

Install Node.js dependencies:

```bash
npm install
```

### 4. Environment Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and configure your settings.

### 5. Running the Application

Start the backend server:

```bash
python neurobridge.py
```

In a separate terminal, start the frontend:

```bash
npm run dev
```

The application will be available at `http://localhost:8080`

## Development

### Running Tests

Python tests:
```bash
pytest tests/
```

Frontend tests:
```bash
npm test
```

### Building for Production

```bash
npm run build
```

## Troubleshooting

### Common Issues

**Port already in use**: Change the port in `.env` or `vite.config.ts`

**Module not found**: Ensure all dependencies are installed correctly

**Python version issues**: Make sure you're using Python 3.9+

## Next Steps

- Read the [API Documentation](./API.md)
- Check out [Examples](./EXAMPLES.md)
- Join our community discussions

## Support

If you encounter any issues, please:
1. Check the [FAQ](./FAQ.md)
2. Search existing GitHub issues
3. Create a new issue with detailed information