# Moderation API

A FastAPI-based content moderation service using BERT for text classification. This service provides an API endpoint that can analyze text content for various types of harmful content, similar to OpenAI's moderation API.

## Features

- Text content moderation using BERT model
- Multiple category detection including:
  - Harassment
  - Hate speech
  - Self-harm
  - Sexual content
  - Violence
- FastAPI-based REST API
- Docker containerization
- GitHub Actions CI/CD pipeline

## Quick Start with Docker Compose

### CPU only
```yaml
version: '3.8'

services:
  moderation-api:
    image: ghcr.io/haouarihk/moderationAPI:main
    ports:
      - 8000:8000
    restart: unless-stopped
```

### GPU Support
```yaml
version: '3.8'

services:
  moderation-api:
    image: ghcr.io/haouarihk/moderationAPI:main
    ports:
      - 8000:8000
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=all
    restart: unless-stopped
```

The API will be available at `http://localhost:8000`

## API Endpoint

### POST /v1/moderations

Analyzes text content for harmful content.

#### Request Body

```json
{
  "input": [
    "Your text to analyze here",
    "Multiple texts can be analyzed at once"
  ]
}
```

#### Response

```json
{
  "id": "modr-xxxxxx",
  "model": "ModerationBERT-En-02",
  "results": [
    {
      "flagged": true,
      "categories": {
        "sexual": false,
        "hate": true,
        "harassment": false,
        "self_harm": false,
        "sexual_minors": false,
        "hate_threatening": false,
        "violence_graphic": false,
        "self_harm_intent": false,
        "self_harm_instructions": false,
        "harassment_threatening": false,
        "violence": false
      },
      "category_scores": {
        "sexual": 0.123,
        "hate": 0.789,
        "harassment": 0.234,
        "self_harm": 0.045,
        "sexual_minors": 0.012,
        "hate_threatening": 0.345,
        "violence_graphic": 0.067,
        "self_harm_intent": 0.089,
        "self_harm_instructions": 0.023,
        "harassment_threatening": 0.156,
        "violence": 0.234
      }
    }
  ]
}
```

## Alternative Installation Methods

### Docker

If you prefer using Docker directly:

```bash
# Pull the image
docker pull ghcr.io/your-username/moderationAPI:main

# Run the container
docker run -p 8000:8000 ghcr.io/your-username/moderationAPI:main
```

### Local Development

For local development, you'll need:

- Python 3.8+
- CUDA-capable GPU (recommended)
- Docker (optional)

1. Clone the repository:
```bash
git clone https://github.com/your-username/moderationAPI.git
cd moderationAPI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Environment Variables

No environment variables are required for basic operation.

## CI/CD

This project uses GitHub Actions for continuous integration and deployment. On every push to the main branch:

1. The Docker image is built
2. The image is pushed to GitHub Container Registry (GHCR)
3. The image is tagged with:
   - `main` for the most recent push
   - Commit SHA for specific versions

## License

[Add your license here]

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 
