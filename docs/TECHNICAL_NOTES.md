# Technical Notes on Survey-to-R Agent Implementation

## API Integration Details

The Survey-to-R Agent application uses the OpenRouter API for AI-powered construct detection, despite some naming conventions that might suggest otherwise.

### Function Naming vs Implementation

- The primary AI function is named `gemini_detect_scales()` in the code
- However, the actual implementation uses OpenRouter API through the OpenAI-compatible interface
- The function is configured to use the endpoint `https://openrouter.ai/api/v1`
- The default model is `openai/gpt-4o-mini`, which is an OpenRouter model identifier

### API Configuration

The application is configured to use OpenRouter API with the following environment variables:
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `OPENROUTER_MODEL`: The model to use (default: `openai/gpt-4o-mini`)

### Why the Confusing Naming?

The function and module names may have been initially developed with Gemini in mind, but the implementation was later updated to use the more flexible OpenRouter API. OpenRouter supports multiple models (OpenAI, Anthropic, and others), which provides users with more flexibility in their AI model choices.

### Supported Models

With OpenRouter, you can use various models including:
- OpenAI models: `openai/gpt-4o-mini`, `openai/gpt-4`, etc.
- Anthropic models: `anthropic/claude-3-opus`, `anthropic/claude-3-sonnet`, etc.
- Google models: `google/gemini-pro`, `google/gemini-flash`, etc.
- And many other providers supported by OpenRouter

This architecture provides flexibility to use different AI providers while maintaining the same API interface.