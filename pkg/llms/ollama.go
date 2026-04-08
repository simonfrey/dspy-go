package llms

import (
	"context"
	"encoding/base64"
	stderrors "errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	ollamaapi "github.com/ollama/ollama/api"
	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
)

// OllamaConfig holds configuration for Ollama provider.
type OllamaConfig struct {
	UseOpenAIAPI bool   `yaml:"use_openai_api" json:"use_openai_api"` // Default: true (modern Ollama)
	BaseURL      string `yaml:"base_url" json:"base_url"`             // Default: http://localhost:11434
	APIKey       string `yaml:"api_key" json:"api_key"`               // Optional for auth
	Timeout      int    `yaml:"timeout" json:"timeout"`               // Default: 60
}

// OllamaLLM implements the core.LLM interface for Ollama-hosted models with
// dual-mode support. In native mode it talks to /api/generate and friends via
// the official github.com/ollama/ollama/api client. In OpenAI-compatible mode
// it talks to /v1/* via the openai-go SDK pointed at the Ollama base URL.
type OllamaLLM struct {
	*core.BaseLLM
	config       OllamaConfig
	nativeClient *ollamaapi.Client // populated when UseOpenAIAPI is false
	openaiClient openaisdk.Client  // populated when UseOpenAIAPI is true
}

// Option pattern for flexible configuration.
type OllamaOption func(*OllamaConfig)

// WithNativeAPI configures Ollama to use native API mode.
func WithNativeAPI() OllamaOption {
	return func(c *OllamaConfig) { c.UseOpenAIAPI = false }
}

// WithOpenAIAPI configures Ollama to use OpenAI-compatible API mode.
func WithOpenAIAPI() OllamaOption {
	return func(c *OllamaConfig) { c.UseOpenAIAPI = true }
}

// WithBaseURL sets the base URL for Ollama.
func WithBaseURL(url string) OllamaOption {
	return func(c *OllamaConfig) { c.BaseURL = url }
}

// WithAuth sets authentication for Ollama (some deployments require it).
func WithAuth(apiKey string) OllamaOption {
	return func(c *OllamaConfig) { c.APIKey = apiKey }
}

// WithTimeout sets the timeout for requests.
func WithTimeout(timeout int) OllamaOption {
	return func(c *OllamaConfig) { c.Timeout = timeout }
}

// NewOllamaLLM creates a new OllamaLLM instance with modern defaults.
func NewOllamaLLM(modelID core.ModelID, options ...OllamaOption) (*OllamaLLM, error) {
	config := OllamaConfig{
		UseOpenAIAPI: true,
		BaseURL:      "http://localhost:11434",
		Timeout:      60,
	}

	for _, option := range options {
		option(&config)
	}

	return newOllamaLLMWithConfig(config, modelID)
}

// NewOllamaLLMFromConfig creates a new OllamaLLM instance from configuration.
func NewOllamaLLMFromConfig(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (*OllamaLLM, error) {
	ollamaConfig, err := parseOllamaConfig(config)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to parse Ollama configuration"),
			errors.Fields{"model": modelID})
	}

	return newOllamaLLMWithConfig(ollamaConfig, modelID)
}

// newOllamaLLMWithConfig creates OllamaLLM with the given configuration.
func newOllamaLLMWithConfig(config OllamaConfig, modelID core.ModelID) (*OllamaLLM, error) {
	modelName := strings.TrimPrefix(string(modelID), "ollama:")
	if modelName == "" {
		return nil, errors.New(errors.InvalidInput, "model name is required")
	}

	var endpointCfg *core.EndpointConfig
	if config.UseOpenAIAPI {
		headers := map[string]string{
			"Content-Type": "application/json",
		}
		if config.APIKey != "" {
			headers["Authorization"] = "Bearer " + config.APIKey
		}
		endpointCfg = &core.EndpointConfig{
			BaseURL:    config.BaseURL,
			Path:       "/v1/chat/completions",
			Headers:    headers,
			TimeoutSec: config.Timeout,
		}
	} else {
		endpointCfg = &core.EndpointConfig{
			BaseURL:    config.BaseURL,
			Path:       "/api/generate",
			Headers:    map[string]string{"Content-Type": "application/json"},
			TimeoutSec: config.Timeout,
		}
	}

	capabilities := []core.Capability{
		core.CapabilityCompletion,
		core.CapabilityChat,
		core.CapabilityJSON,
	}
	if supportsOllamaStreaming(modelName) {
		capabilities = append(capabilities, core.CapabilityStreaming)
	}
	if config.UseOpenAIAPI || supportsOllamaEmbedding(modelName) {
		capabilities = append(capabilities, core.CapabilityEmbedding)
	}
	if config.UseOpenAIAPI {
		capabilities = append(capabilities, core.CapabilityToolCalling)
	}

	httpClient := &http.Client{Timeout: time.Duration(config.Timeout) * time.Second}

	llm := &OllamaLLM{
		BaseLLM: core.NewBaseLLM("ollama", core.ModelID(modelName), capabilities, endpointCfg),
		config:  config,
	}

	if config.UseOpenAIAPI {
		llm.openaiClient = buildOllamaOpenAISDKClient(config, httpClient)
	} else {
		baseURL, err := url.Parse(config.BaseURL)
		if err != nil {
			return nil, errors.WithFields(
				errors.Wrap(err, errors.InvalidInput, "failed to parse Ollama base URL"),
				errors.Fields{"base_url": config.BaseURL})
		}
		llm.nativeClient = ollamaapi.NewClient(baseURL, httpClient)
	}

	return llm, nil
}

// buildOllamaOpenAISDKClient constructs an openai-go SDK client pointed at an
// Ollama server's OpenAI-compatible /v1/ endpoint.
func buildOllamaOpenAISDKClient(config OllamaConfig, httpClient *http.Client) openaisdk.Client {
	base := strings.TrimRight(config.BaseURL, "/") + "/v1/"

	apiKey := config.APIKey
	if apiKey == "" {
		// Many Ollama deployments don't require auth. Provide a placeholder
		// so the SDK doesn't pick up an OPENAI_API_KEY from the environment.
		apiKey = "ollama"
	}

	opts := []option.RequestOption{
		option.WithBaseURL(base),
		option.WithMaxRetries(0),
		option.WithHTTPClient(httpClient),
		option.WithAPIKey(apiKey),
	}
	return openaisdk.NewClient(opts...)
}

// parseOllamaConfig parses configuration supporting both legacy and modern formats.
func parseOllamaConfig(config core.ProviderConfig) (OllamaConfig, error) {
	result := OllamaConfig{
		UseOpenAIAPI: true,
		BaseURL:      "http://localhost:11434",
		Timeout:      60,
	}

	if config.BaseURL != "" {
		result.BaseURL = config.BaseURL
	}
	if config.Endpoint != nil && config.Endpoint.BaseURL != "" {
		result.BaseURL = config.Endpoint.BaseURL
	}

	if config.APIKey != "" {
		result.APIKey = config.APIKey
	}

	if config.Endpoint != nil && config.Endpoint.TimeoutSec > 0 {
		result.Timeout = config.Endpoint.TimeoutSec
	}

	if config.Params != nil {
		if useOpenAI, ok := config.Params["use_openai_api"].(bool); ok {
			result.UseOpenAIAPI = useOpenAI
		}
		if timeout, ok := config.Params["timeout"].(int); ok {
			result.Timeout = timeout
		}
	}

	return result, nil
}

// Generate implements the core.LLM interface with dual-mode support.
func (o *OllamaLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	if o.config.UseOpenAIAPI {
		return o.generateOpenAI(ctx, prompt, options...)
	}
	return o.generateNative(ctx, prompt, options...)
}

// generateOpenAI uses the openai-go SDK pointed at the Ollama OpenAI-compatible endpoint.
func (o *OllamaLLM) generateOpenAI(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	params := openaisdk.ChatCompletionNewParams{
		Model: shared.ChatModel(o.ModelID()),
		Messages: []openaisdk.ChatCompletionMessageParamUnion{
			openaisdk.UserMessage(prompt),
		},
	}
	applyOpenAIGenerateOptions(&params, o.ModelID(), opts)

	response, err := o.openaiClient.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, mapOllamaOpenAIError(err, o.ModelID())
	}

	if len(response.Choices) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "no choices in response"),
			errors.Fields{"model": o.ModelID()})
	}

	return &core.LLMResponse{
		Content: response.Choices[0].Message.Content,
		Usage: &core.TokenInfo{
			PromptTokens:     int(response.Usage.PromptTokens),
			CompletionTokens: int(response.Usage.CompletionTokens),
			TotalTokens:      int(response.Usage.TotalTokens),
		},
		Metadata: map[string]interface{}{
			"model": response.Model,
			"mode":  "openai",
		},
	}, nil
}

// generateNative uses the official github.com/ollama/ollama/api client.
func (o *OllamaLLM) generateNative(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	streamFalse := false
	req := &ollamaapi.GenerateRequest{
		Model:   string(o.ModelID()),
		Prompt:  prompt,
		Stream:  &streamFalse,
		Options: buildOllamaNativeOptions(opts),
	}

	var lastResp ollamaapi.GenerateResponse
	var received bool
	err := o.nativeClient.Generate(ctx, req, func(r ollamaapi.GenerateResponse) error {
		lastResp = r
		received = true
		return nil
	})
	if err != nil {
		return nil, mapOllamaNativeError(err, o.ModelID())
	}
	if !received {
		return nil, errors.WithFields(
			errors.New(errors.LLMGenerationFailed, "ollama native API returned no response"),
			errors.Fields{"model": o.ModelID()})
	}

	return &core.LLMResponse{
		Content:       lastResp.Response,
		ContentBlocks: ollamaContentBlocksFromGenerateResponse(lastResp),
		Metadata: map[string]interface{}{
			"model": lastResp.Model,
			"mode":  "native",
		},
	}, nil
}

// ollamaContentBlocksFromGenerateResponse extracts text and (experimental)
// image output from an Ollama /api/generate response. The Ollama SDK exposes
// a base64-encoded Image field on GenerateResponse for image-generation models;
// when present we decode it and emit an image ContentBlock alongside the text.
func ollamaContentBlocksFromGenerateResponse(resp ollamaapi.GenerateResponse) []core.ContentBlock {
	var blocks []core.ContentBlock
	if resp.Response != "" {
		blocks = append(blocks, core.NewTextBlock(resp.Response))
	}
	if resp.Image != "" {
		if data, err := base64.StdEncoding.DecodeString(resp.Image); err == nil && len(data) > 0 {
			blocks = append(blocks, core.NewImageBlock(data, "image/png"))
		}
	}
	if len(blocks) == 0 {
		return nil
	}
	return blocks
}

// StreamGenerate implements streaming with dual-mode support.
func (o *OllamaLLM) StreamGenerate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	if o.config.UseOpenAIAPI {
		return o.streamGenerateOpenAI(ctx, prompt, options...)
	}
	return o.streamGenerateNative(ctx, prompt, options...)
}

func (o *OllamaLLM) streamGenerateOpenAI(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	params := openaisdk.ChatCompletionNewParams{
		Model: shared.ChatModel(o.ModelID()),
		Messages: []openaisdk.ChatCompletionMessageParamUnion{
			openaisdk.UserMessage(prompt),
		},
	}
	applyOpenAIGenerateOptions(&params, o.ModelID(), opts)

	chunkChan := make(chan core.StreamChunk)
	streamCtx, cancelStream := context.WithCancel(ctx)

	go func() {
		defer close(chunkChan)
		defer cancelStream()

		stream := o.openaiClient.Chat.Completions.NewStreaming(streamCtx, params)
		defer func() {
			_ = stream.Close()
		}()

		pumpOpenAIChatCompletionStream(streamCtx, stream, chunkChan)
	}()

	return &core.StreamResponse{
		ChunkChannel: chunkChan,
		Cancel:       cancelStream,
	}, nil
}

func (o *OllamaLLM) streamGenerateNative(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	streamTrue := true
	req := &ollamaapi.GenerateRequest{
		Model:   string(o.ModelID()),
		Prompt:  prompt,
		Stream:  &streamTrue,
		Options: buildOllamaNativeOptions(opts),
	}

	chunkChan := make(chan core.StreamChunk)
	streamCtx, cancelStream := context.WithCancel(ctx)

	go func() {
		defer close(chunkChan)

		err := o.nativeClient.Generate(streamCtx, req, func(r ollamaapi.GenerateResponse) error {
			if err := streamCtx.Err(); err != nil {
				return err
			}
			if r.Response != "" {
				select {
				case chunkChan <- core.StreamChunk{Content: r.Response}:
				case <-streamCtx.Done():
					return streamCtx.Err()
				}
			}
			if r.Done {
				select {
				case chunkChan <- core.StreamChunk{Done: true}:
				case <-streamCtx.Done():
					return streamCtx.Err()
				}
			}
			return nil
		})
		if err != nil && !stderrors.Is(err, context.Canceled) {
			select {
			case chunkChan <- core.StreamChunk{Error: mapOllamaNativeError(err, o.ModelID())}:
			case <-streamCtx.Done():
			}
		}
	}()

	return &core.StreamResponse{
		ChunkChannel: chunkChan,
		Cancel:       cancelStream,
	}, nil
}

// CreateEmbedding implements embedding generation with OpenAI-compatible mode support.
func (o *OllamaLLM) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	if !o.config.UseOpenAIAPI && !supportsOllamaEmbedding(o.ModelID()) {
		return nil, errors.WithFields(
			errors.New(errors.UnsupportedOperation, "embeddings require OpenAI API mode or embedding model"),
			errors.Fields{
				"provider":       "ollama",
				"use_openai_api": o.config.UseOpenAIAPI,
				"model":          o.ModelID(),
			})
	}

	if o.config.UseOpenAIAPI {
		return o.createEmbeddingOpenAI(ctx, input, options...)
	}
	return o.createEmbeddingNative(ctx, input, options...)
}

func (o *OllamaLLM) createEmbeddingOpenAI(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	opts := core.NewEmbeddingOptions()
	for _, opt := range options {
		opt(opts)
	}

	model := string(o.ModelID())
	if opts.Model != "" {
		model = opts.Model
	}

	params := openaisdk.EmbeddingNewParams{
		Input: openaisdk.EmbeddingNewParamsInputUnion{
			OfString: param.NewOpt(input),
		},
		Model:          openaisdk.EmbeddingModel(model),
		EncodingFormat: openaisdk.EmbeddingNewParamsEncodingFormatFloat,
	}

	response, err := o.openaiClient.Embeddings.New(ctx, params)
	if err != nil {
		return nil, mapOllamaOpenAIError(err, o.ModelID())
	}

	if len(response.Data) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "no embeddings in response"),
			errors.Fields{"model": o.ModelID()})
	}

	embedding := make([]float32, len(response.Data[0].Embedding))
	for i, v := range response.Data[0].Embedding {
		embedding[i] = float32(v)
	}

	return &core.EmbeddingResult{
		Vector:     embedding,
		TokenCount: int(response.Usage.TotalTokens),
		Metadata: map[string]interface{}{
			"model": response.Model,
			"mode":  "openai",
		},
	}, nil
}

func (o *OllamaLLM) createEmbeddingNative(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	opts := core.NewEmbeddingOptions()
	for _, opt := range options {
		opt(opts)
	}

	req := &ollamaapi.EmbeddingRequest{
		Model:   string(o.ModelID()),
		Prompt:  input,
		Options: opts.Params,
	}

	resp, err := o.nativeClient.Embeddings(ctx, req)
	if err != nil {
		return nil, mapOllamaNativeError(err, o.ModelID())
	}

	embedding := make([]float32, len(resp.Embedding))
	for i, v := range resp.Embedding {
		embedding[i] = float32(v)
	}

	return &core.EmbeddingResult{
		Vector: embedding,
		Metadata: map[string]interface{}{
			"model": o.ModelID(),
			"mode":  "native",
		},
	}, nil
}

// GenerateWithJSON implements JSON mode generation.
func (o *OllamaLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	response, err := o.Generate(ctx, prompt, options...)
	if err != nil {
		return nil, err
	}

	return utils.ParseJSONResponse(response.Content)
}

// GenerateWithFunctions performs function calling via the OpenAI-compatible mode.
func (o *OllamaLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
	if !o.config.UseOpenAIAPI {
		return nil, errors.WithFields(
			errors.New(errors.UnsupportedOperation, "function calling requires OpenAI-compatible API mode for Ollama"),
			errors.Fields{
				"provider":       "ollama",
				"model":          o.ModelID(),
				"use_openai_api": false,
			})
	}

	if len(functions) == 0 {
		return nil, errors.New(errors.InvalidInput, "at least one function schema is required")
	}

	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	tools, err := convertOpenAIFunctionSchemasToSDK(functions)
	if err != nil {
		return nil, err
	}

	params := openaisdk.ChatCompletionNewParams{
		Model: shared.ChatModel(o.ModelID()),
		Messages: []openaisdk.ChatCompletionMessageParamUnion{
			openaisdk.UserMessage(prompt),
		},
		Tools: tools,
		ToolChoice: openaisdk.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: param.NewOpt("auto"),
		},
	}
	applyOpenAIGenerateOptions(&params, o.ModelID(), opts)

	response, err := o.openaiClient.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, mapOllamaOpenAIError(err, o.ModelID())
	}

	if len(response.Choices) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "no choices in response"),
			errors.Fields{"model": o.ModelID()})
	}

	return buildOpenAIToolResultFromSDK(response.Choices[0], response.Usage, "ollama")
}

// GenerateWithContent implements multimodal content generation. Image inputs
// are forwarded to the model as part of a single user turn; in OpenAI mode the
// SDK serializes them as image_url data URLs, and in native mode they are
// passed through ollamaapi.ChatRequest as base-64 image bytes. Inputs are not
// stored or used for training, identical to text prompts.
func (o *OllamaLLM) GenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.LLMResponse, error) {
	if len(content) == 0 {
		return nil, errors.New(errors.InvalidInput, "no content provided")
	}
	if !ollamaContentHasNonText(content) {
		return o.Generate(ctx, ollamaJoinTextBlocks(content), options...)
	}
	if o.config.UseOpenAIAPI {
		return o.generateWithContentOpenAI(ctx, content, options...)
	}
	return o.generateWithContentNative(ctx, content, options...)
}

// StreamGenerateWithContent implements multimodal streaming content generation.
func (o *OllamaLLM) StreamGenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.StreamResponse, error) {
	if len(content) == 0 {
		return nil, errors.New(errors.InvalidInput, "no content provided")
	}
	if !ollamaContentHasNonText(content) {
		return o.StreamGenerate(ctx, ollamaJoinTextBlocks(content), options...)
	}
	if o.config.UseOpenAIAPI {
		return o.streamGenerateWithContentOpenAI(ctx, content, options...)
	}
	return o.streamGenerateWithContentNative(ctx, content, options...)
}

func (o *OllamaLLM) generateWithContentOpenAI(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.LLMResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	parts, err := convertContentBlocksToOpenAIParts(content)
	if err != nil {
		return nil, err
	}
	if len(parts) == 0 {
		return nil, errors.New(errors.InvalidInput, "no content provided")
	}

	params := openaisdk.ChatCompletionNewParams{
		Model: shared.ChatModel(o.ModelID()),
		Messages: []openaisdk.ChatCompletionMessageParamUnion{
			openaiUserMessageFromParts(parts),
		},
	}
	applyOpenAIGenerateOptions(&params, o.ModelID(), opts)

	response, err := o.openaiClient.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, mapOllamaOpenAIError(err, o.ModelID())
	}
	if len(response.Choices) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "no choices in response"),
			errors.Fields{"model": o.ModelID()})
	}

	return &core.LLMResponse{
		Content: response.Choices[0].Message.Content,
		Usage: &core.TokenInfo{
			PromptTokens:     int(response.Usage.PromptTokens),
			CompletionTokens: int(response.Usage.CompletionTokens),
			TotalTokens:      int(response.Usage.TotalTokens),
		},
		Metadata: map[string]interface{}{
			"model": response.Model,
			"mode":  "openai",
		},
	}, nil
}

func (o *OllamaLLM) streamGenerateWithContentOpenAI(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.StreamResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	parts, err := convertContentBlocksToOpenAIParts(content)
	if err != nil {
		return nil, err
	}
	if len(parts) == 0 {
		return nil, errors.New(errors.InvalidInput, "no content provided")
	}

	params := openaisdk.ChatCompletionNewParams{
		Model: shared.ChatModel(o.ModelID()),
		Messages: []openaisdk.ChatCompletionMessageParamUnion{
			openaiUserMessageFromParts(parts),
		},
	}
	applyOpenAIGenerateOptions(&params, o.ModelID(), opts)

	chunkChan := make(chan core.StreamChunk)
	streamCtx, cancelStream := context.WithCancel(ctx)

	go func() {
		defer close(chunkChan)
		defer cancelStream()

		stream := o.openaiClient.Chat.Completions.NewStreaming(streamCtx, params)
		defer func() {
			_ = stream.Close()
		}()

		pumpOpenAIChatCompletionStream(streamCtx, stream, chunkChan)
	}()

	return &core.StreamResponse{
		ChunkChannel: chunkChan,
		Cancel:       cancelStream,
	}, nil
}

func (o *OllamaLLM) generateWithContentNative(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.LLMResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	message := buildOllamaNativeUserMessage(content)
	streamFalse := false
	req := &ollamaapi.ChatRequest{
		Model:    string(o.ModelID()),
		Messages: []ollamaapi.Message{message},
		Stream:   &streamFalse,
		Options:  buildOllamaNativeOptions(opts),
	}

	var lastResp ollamaapi.ChatResponse
	var received bool
	err := o.nativeClient.Chat(ctx, req, func(r ollamaapi.ChatResponse) error {
		lastResp = r
		received = true
		return nil
	})
	if err != nil {
		return nil, mapOllamaNativeError(err, o.ModelID())
	}
	if !received {
		return nil, errors.WithFields(
			errors.New(errors.LLMGenerationFailed, "ollama native API returned no response"),
			errors.Fields{"model": o.ModelID()})
	}

	return &core.LLMResponse{
		Content:       lastResp.Message.Content,
		ContentBlocks: ollamaContentBlocksFromMessage(lastResp.Message),
		Metadata: map[string]interface{}{
			"model": lastResp.Model,
			"mode":  "native",
		},
	}, nil
}

// ollamaContentBlocksFromMessage extracts text and image output from an Ollama
// native API message into core.ContentBlock slices. Image bytes are wrapped as
// FieldTypeImage blocks; the text content is also included so callers iterating
// ContentBlocks see the full response.
func ollamaContentBlocksFromMessage(msg ollamaapi.Message) []core.ContentBlock {
	var blocks []core.ContentBlock
	if msg.Content != "" {
		blocks = append(blocks, core.NewTextBlock(msg.Content))
	}
	for _, img := range msg.Images {
		if len(img) == 0 {
			continue
		}
		blocks = append(blocks, core.NewImageBlock([]byte(img), "image/png"))
	}
	if len(blocks) == 0 {
		return nil
	}
	return blocks
}

func (o *OllamaLLM) streamGenerateWithContentNative(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.StreamResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	message := buildOllamaNativeUserMessage(content)
	streamTrue := true
	req := &ollamaapi.ChatRequest{
		Model:    string(o.ModelID()),
		Messages: []ollamaapi.Message{message},
		Stream:   &streamTrue,
		Options:  buildOllamaNativeOptions(opts),
	}

	chunkChan := make(chan core.StreamChunk)
	streamCtx, cancelStream := context.WithCancel(ctx)

	go func() {
		defer close(chunkChan)

		err := o.nativeClient.Chat(streamCtx, req, func(r ollamaapi.ChatResponse) error {
			if err := streamCtx.Err(); err != nil {
				return err
			}
			if r.Message.Content != "" {
				select {
				case chunkChan <- core.StreamChunk{Content: r.Message.Content}:
				case <-streamCtx.Done():
					return streamCtx.Err()
				}
			}
			if r.Done {
				select {
				case chunkChan <- core.StreamChunk{Done: true}:
				case <-streamCtx.Done():
					return streamCtx.Err()
				}
			}
			return nil
		})
		if err != nil && !stderrors.Is(err, context.Canceled) {
			select {
			case chunkChan <- core.StreamChunk{Error: mapOllamaNativeError(err, o.ModelID())}:
			case <-streamCtx.Done():
			}
		}
	}()

	return &core.StreamResponse{
		ChunkChannel: chunkChan,
		Cancel:       cancelStream,
	}, nil
}

// buildOllamaNativeUserMessage flattens text blocks into a single Content
// string and gathers image blocks as raw byte slices on the Images field, the
// shape Ollama's native /api/chat endpoint expects. Audio is not supported by
// Ollama natively and is dropped with a textual placeholder.
func buildOllamaNativeUserMessage(content []core.ContentBlock) ollamaapi.Message {
	var textParts []string
	var images []ollamaapi.ImageData
	for _, block := range content {
		switch block.Type {
		case core.FieldTypeText:
			if block.Text != "" {
				textParts = append(textParts, block.Text)
			}
		case core.FieldTypeImage:
			if len(block.Data) > 0 {
				img := make([]byte, len(block.Data))
				copy(img, block.Data)
				images = append(images, ollamaapi.ImageData(img))
			}
		case core.FieldTypeAudio:
			textParts = append(textParts, fmt.Sprintf("[Audio: %s]", block.MimeType))
		}
	}
	return ollamaapi.Message{
		Role:    "user",
		Content: strings.Join(textParts, "\n"),
		Images:  images,
	}
}

func ollamaContentHasNonText(blocks []core.ContentBlock) bool {
	for _, block := range blocks {
		if block.Type != core.FieldTypeText {
			return true
		}
	}
	return false
}

func ollamaJoinTextBlocks(blocks []core.ContentBlock) string {
	var parts []string
	for _, block := range blocks {
		if block.Type == core.FieldTypeText && block.Text != "" {
			parts = append(parts, block.Text)
		}
	}
	return strings.TrimSpace(strings.Join(parts, "\n"))
}

// CreateEmbeddings generates embeddings for multiple inputs.
func (o *OllamaLLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	if o.config.UseOpenAIAPI {
		opts := core.NewEmbeddingOptions()
		for _, opt := range options {
			opt(opts)
		}

		model := string(o.ModelID())
		if opts.Model != "" {
			model = opts.Model
		}

		params := openaisdk.EmbeddingNewParams{
			Input: openaisdk.EmbeddingNewParamsInputUnion{
				OfArrayOfStrings: inputs,
			},
			Model:          openaisdk.EmbeddingModel(model),
			EncodingFormat: openaisdk.EmbeddingNewParamsEncodingFormatFloat,
		}

		response, err := o.openaiClient.Embeddings.New(ctx, params)
		if err != nil {
			return nil, mapOllamaOpenAIError(err, o.ModelID())
		}

		if len(response.Data) == 0 {
			return nil, errors.New(errors.InvalidResponse, "no embeddings in response")
		}

		results := make([]core.EmbeddingResult, len(response.Data))
		for i, data := range response.Data {
			embedding := make([]float32, len(data.Embedding))
			for j, v := range data.Embedding {
				embedding[j] = float32(v)
			}
			perInputTokens := 0
			if len(inputs) > 0 {
				perInputTokens = int(response.Usage.TotalTokens) / len(inputs)
			}
			results[i] = core.EmbeddingResult{
				Vector:     embedding,
				TokenCount: perInputTokens,
				Metadata: map[string]interface{}{
					"model":       response.Model,
					"mode":        "openai",
					"batch_index": i,
				},
			}
		}

		return &core.BatchEmbeddingResult{
			Embeddings: results,
			ErrorIndex: -1,
		}, nil
	}

	// Native mode: loop over inputs since the legacy /api/embeddings endpoint
	// is single-input. Behavior matches the prior implementation.
	var allResults []core.EmbeddingResult
	var firstError error
	var errorIndex = -1

	for i, input := range inputs {
		result, err := o.CreateEmbedding(ctx, input, options...)
		if err != nil {
			if firstError == nil {
				firstError = err
				errorIndex = i
			}
			continue
		}

		if result.Metadata == nil {
			result.Metadata = make(map[string]interface{})
		}
		result.Metadata["batch_index"] = i

		allResults = append(allResults, *result)
	}

	return &core.BatchEmbeddingResult{
		Embeddings: allResults,
		Error:      firstError,
		ErrorIndex: errorIndex,
	}, firstError
}

// buildOllamaNativeOptions translates core.GenerateOptions into the
// model-options map shape Ollama's /api/generate expects (e.g. "num_predict",
// "temperature").
func buildOllamaNativeOptions(opts *core.GenerateOptions) map[string]any {
	if opts == nil {
		return nil
	}
	out := map[string]any{}
	if opts.MaxTokens > 0 {
		out["num_predict"] = opts.MaxTokens
	}
	if opts.Temperature >= 0 {
		out["temperature"] = opts.Temperature
	}
	if opts.TopP > 0 {
		out["top_p"] = opts.TopP
	}
	if len(opts.Stop) > 0 {
		out["stop"] = opts.Stop
	}
	if opts.PresencePenalty != 0 {
		out["presence_penalty"] = opts.PresencePenalty
	}
	if opts.FrequencyPenalty != 0 {
		out["frequency_penalty"] = opts.FrequencyPenalty
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// mapOllamaOpenAIError shapes an openai-go SDK error into a dspy-go errors.Error
// and ensures the message contains the HTTP status code (the historical contract
// the Ollama tests assert against).
func mapOllamaOpenAIError(err error, model string) error {
	if err == nil {
		return nil
	}
	var apiErr *openaisdk.Error
	if stderrors.As(err, &apiErr) {
		msg := strings.TrimSpace(apiErr.Message)
		if msg == "" {
			msg = fmt.Sprintf("API request failed with status %d", apiErr.StatusCode)
		}
		return errors.WithFields(
			errors.New(errors.LLMGenerationFailed, msg),
			errors.Fields{
				"model":       model,
				"status_code": apiErr.StatusCode,
				"type":        apiErr.Type,
				"code":        apiErr.Code,
			})
	}
	return errors.WithFields(
		errors.Wrap(err, errors.LLMGenerationFailed, "request failed"),
		errors.Fields{"model": model})
}

// mapOllamaNativeError shapes an ollama-api SDK error into a dspy-go errors.Error.
func mapOllamaNativeError(err error, model string) error {
	if err == nil {
		return nil
	}
	var statusErr ollamaapi.StatusError
	if stderrors.As(err, &statusErr) {
		msg := fmt.Sprintf("API request failed with status code %d", statusErr.StatusCode)
		if trimmed := strings.TrimSpace(statusErr.ErrorMessage); trimmed != "" {
			msg = fmt.Sprintf("%s: %s", msg, trimmed)
		}
		return errors.WithFields(
			errors.New(errors.LLMGenerationFailed, msg),
			errors.Fields{"model": model, "status_code": statusErr.StatusCode})
	}
	return errors.WithFields(
		errors.Wrap(err, errors.LLMGenerationFailed, "request failed"),
		errors.Fields{"model": model})
}

// supportsOllamaStreaming checks if the model supports streaming.
// Currently all Ollama models support streaming, but this function is kept
// as a placeholder for future model-specific capability checks.
// The modelName parameter is unused but kept for API consistency.
func supportsOllamaStreaming(_ string) bool {
	return true
}

// supportsOllamaEmbedding checks if the model supports embedding.
func supportsOllamaEmbedding(modelName string) bool {
	embeddingModels := []string{
		"nomic-embed-text",
		"mxbai-embed-large",
		"snowflake-arctic-embed",
		"all-minilm",
	}

	modelLower := strings.ToLower(modelName)
	for _, embeddingModel := range embeddingModels {
		if strings.Contains(modelLower, embeddingModel) {
			return true
		}
	}

	if strings.Contains(modelLower, "embed") {
		return true
	}

	return false
}

// OllamaProviderFactory creates OllamaLLM instances.
func OllamaProviderFactory(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	return NewOllamaLLMFromConfig(ctx, config, modelID)
}
