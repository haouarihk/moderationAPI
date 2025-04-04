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
- Docker containerization with both CPU and GPU support
- GitHub Actions CI/CD pipeline

## Model Information

This service uses the `ifmain/ModerationBERT-En-02` model, which is a fine-tuned BERT model specifically designed for content moderation. The model can detect and classify content across 11 different categories:

- Harassment
- Harassment/Threatening
- Hate
- Hate/Threatening
- Self-harm
- Self-harm/Instructions
- Self-harm/Intent
- Sexual
- Sexual/Minors
- Violence
- Violence/Graphic

The model outputs both binary classifications (flagged/not flagged) and confidence scores for each category, allowing for fine-grained content analysis.

## Quick Start with Docker Compose

### CPU only
```yaml
version: '3.8'

services:
  moderation-api:
    image: ghcr.io/haouarihk/moderationapi:main-cpu
    ports:
      - 8000:8000
    restart: unless-stopped
```

### GPU Support
```yaml
version: '3.8'

services:
  moderation-api:
    image: ghcr.io/haouarihk/moderationapi:main
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
# Pull the CPU image
docker pull ghcr.io/haouarihk/moderationapi:latest-cpu

# Run the container
docker run -p 8000:8000 ghcr.io/haouarihk/moderationapi:latest-cpu
```

### Local Development

For local development, you'll need:

- Python 3.8+
- CUDA-capable GPU (recommended)
- Docker (optional)

1. Clone the repository:
```bash
git clone https://github.com/haouarihk/moderationAPI.git
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

This project uses separate GitHub Actions workflows for CPU and GPU images:

1. GPU Image Workflow (`.github/workflows/docker-build-gpu.yml`):
   - Builds and pushes the GPU-optimized image
   - Tagged as `latest`
   - Uses CUDA base image

2. CPU Image Workflow (`.github/workflows/docker-build-cpu.yml`):
   - Builds and pushes the CPU-optimized image
   - Tagged as `latest-cpu`
   - Uses slim base image

Both workflows:
- Trigger on pushes to the main branch
- Push to GitHub Container Registry (GHCR)
- Include commit SHA tags for specific versions

## License

[Add your license here]

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 
