package llms

import (
	"context"
	"encoding/base64"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"

	"google.golang.org/genai"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
)

// GeminiLLM implements the core.LLM interface for Google's Gemini model.
type GeminiLLM struct {
	*core.BaseLLM
	apiKey string

	// SDK client. Populated eagerly by the public constructors and lazily by
	// ensureClient when callers (notably tests) build a GeminiLLM struct
	// directly without going through a constructor.
	clientMu  sync.Mutex
	client    *genai.Client
	clientErr error
}

// GeminiRequest represents the request structure for Gemini API.
type geminiRequest struct {
	Contents         []geminiContent        `json:"contents"`
	GenerationConfig geminiGenerationConfig `json:"generationConfig,omitempty"`
	Tools            []geminiTool           `json:"tools,omitempty"`
	ToolConfig       *geminiToolConfig      `json:"toolConfig,omitempty"`
}

// Add these new types to support function calling.
type geminiTool struct {
	FunctionDeclarations []geminiFunctionDeclaration `json:"functionDeclarations"`
}

type geminiToolConfig struct {
	FunctionCallingConfig *geminiFunctionCallingConfig `json:"functionCallingConfig,omitempty"`
}

type geminiFunctionCallingConfig struct {
	Mode                 string   `json:"mode,omitempty"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

type geminiFunctionDeclaration struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters"`
}
type geminiContent struct {
	Parts []geminiPart `json:"parts"`
	Role  string       `json:"role,omitempty"`
}

type geminiPart struct {
	Text             string                      `json:"text,omitempty"`
	InlineData       *geminiInlineData           `json:"inlineData,omitempty"`
	FileData         *geminiFileData             `json:"fileData,omitempty"`
	FunctionCall     *geminiFunctionCall         `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponsePart `json:"functionResponse,omitempty"`
	Thought          bool                        `json:"thought,omitempty"`
	ThoughtSignature string                      `json:"thoughtSignature,omitempty"`
}

type geminiFunctionResponsePart struct {
	ID       string                 `json:"id,omitempty"`
	Name     string                 `json:"name"`
	Response map[string]interface{} `json:"response"`
}

// geminiInlineData represents inline binary data (base64 encoded).
type geminiInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"` // base64 encoded
}

// geminiFileData represents a file uploaded to Gemini (for large files).
type geminiFileData struct {
	MimeType string `json:"mimeType"`
	FileURI  string `json:"fileUri"`
}

type geminiGenerationConfig struct {
	Temperature        float64               `json:"temperature,omitempty"`
	MaxOutputTokens    int                   `json:"maxOutputTokens,omitempty"`
	TopP               float64               `json:"topP,omitempty"`
	ThinkingConfig     *geminiThinkingConfig `json:"thinkingConfig,omitempty"`
	ResponseModalities []string              `json:"responseModalities,omitempty"`
}

type geminiThinkingConfig struct {
	IncludeThoughts bool `json:"includeThoughts,omitempty"`
}

// GeminiResponse represents the response structure from Gemini API.
type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []geminiPart `json:"parts"`
		} `json:"content"`
		FinishReason string `json:"finishReason,omitempty"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
		ThoughtsTokenCount   int `json:"thoughtsTokenCount"`
	} `json:"usageMetadata"`
}

type geminiFunctionCallResponse struct {
	Candidates []struct {
		Content struct {
			Parts []geminiPart `json:"parts"`
		} `json:"content"`
		FinishReason string `json:"finishReason,omitempty"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
		ThoughtsTokenCount   int `json:"thoughtsTokenCount"`
	} `json:"usageMetadata"`
	PromptFeedback map[string]any `json:"promptFeedback,omitempty"`
}
type geminiFunctionCall struct {
	ID        string                 `json:"id,omitempty"`
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"args"`
}

// Request and response structures for Gemini embeddings.
type geminiEmbeddingRequest struct {
	Model   string `json:"model"`
	Content struct {
		Parts []struct {
			Text string `json:"text"`
		} `json:"parts"`
	} `json:"content"`
	// Task type helps the model generate appropriate embeddings
	TaskType string `json:"taskType,omitempty"`
	// Additional configuration
	Title      string                 `json:"title,omitempty"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
}

type geminiBatchEmbeddingRequest struct {
	Model    string `json:"model"`
	Requests []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"requests"`
	TaskType   string                 `json:"taskType,omitempty"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
}

type geminiEmbeddingResponse struct {
	Embedding struct {
		Values []float32 `json:"values"`
		// Statistics about the generated embedding
		Statistics struct {
			TruncatedInputTokenCount int `json:"truncatedInputTokenCount"`
			TokenCount               int `json:"tokenCount"`
		} `json:"statistics"`
	} `json:"embedding"`
	// Usage information
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}

type geminiBatchEmbeddingResponse struct {
	Embeddings []geminiEmbeddingResponse `json:"embeddings"`
}

// NewGeminiLLM creates a new GeminiLLM instance.
func NewGeminiLLM(apiKey string, model core.ModelID) (*GeminiLLM, error) {
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY") // or whatever env var name you prefer
		if apiKey == "" {
			return nil, errors.New(errors.InvalidInput, "API key is required")
		}

	}

	if model == "" {
		model = core.ModelGoogleGeminiFlash // Default model
	}
	capabilities := []core.Capability{
		core.CapabilityCompletion,
		core.CapabilityChat,
		core.CapabilityJSON,
		core.CapabilityEmbedding,
		core.CapabilityMultimodal,
		core.CapabilityVision,
		core.CapabilityAudio,
	}
	if model == core.ModelGoogleGeminiFlashImage {
		capabilities = append(capabilities, core.CapabilityImageGeneration)
	}
	// Validate model ID
	switch model {
	case core.ModelGoogleGeminiPro, core.ModelGoogleGeminiFlash, core.ModelGoogleGeminiFlashLite,
		core.ModelGoogleGemini3ProPreview, core.ModelGoogleGemini3FlashPreview,
		core.ModelGoogleGemini20Flash, core.ModelGoogleGemini20FlashLite,
		core.ModelGoogleGeminiFlashImage:
		break
	default:
		return nil, errors.WithFields(
			errors.New(errors.InvalidInput, fmt.Sprintf("unsupported Gemini model: %s", model)),
			errors.Fields{"model": model})
	}
	endpoint := &core.EndpointConfig{
		BaseURL: "https://generativelanguage.googleapis.com/v1beta",
		Path:    fmt.Sprintf("/models/%s:generateContent", model),
		Headers: map[string]string{
			"Content-Type": "application/json",
		},
		TimeoutSec: 10 * 60,
	}

	llm := &GeminiLLM{
		apiKey:  apiKey,
		BaseLLM: core.NewBaseLLM("google", model, capabilities, endpoint),
	}
	// Best-effort eager client init. If it fails (e.g. malformed env), the
	// error is surfaced from ensureClient on first use.
	_, _ = llm.ensureClient(context.Background())
	return llm, nil
}

// NewGeminiLLMFromConfig creates a new GeminiLLM instance from configuration.
func NewGeminiLLMFromConfig(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (*GeminiLLM, error) {
	// Get API key from config or environment
	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if apiKey == "" {
		return nil, errors.New(errors.InvalidInput, "API key is required")
	}

	// Use default model if none specified
	if modelID == "" {
		modelID = core.ModelGoogleGeminiFlash
	}

	// Validate model ID
	if !isValidGeminiModel(modelID) {
		return nil, errors.WithFields(
			errors.New(errors.InvalidInput, "unsupported Gemini model"),
			errors.Fields{"model": modelID})
	}

	// Create endpoint configuration
	baseURL := "https://generativelanguage.googleapis.com/v1beta"
	if config.BaseURL != "" {
		baseURL = config.BaseURL
	}

	endpoint := &core.EndpointConfig{
		BaseURL: baseURL,
		Path:    fmt.Sprintf("/models/%s:generateContent", modelID),
		Headers: map[string]string{
			"Content-Type": "application/json",
		},
		TimeoutSec: 10 * 60,
	}

	// Override with config endpoint if provided
	if config.Endpoint != nil {
		if config.Endpoint.BaseURL != "" {
			endpoint.BaseURL = config.Endpoint.BaseURL
		}
		if config.Endpoint.TimeoutSec > 0 {
			endpoint.TimeoutSec = config.Endpoint.TimeoutSec
		}
		for k, v := range config.Endpoint.Headers {
			endpoint.Headers[k] = v
		}
	}

	// Set capabilities
	capabilities := []core.Capability{
		core.CapabilityCompletion,
		core.CapabilityChat,
		core.CapabilityJSON,
		core.CapabilityEmbedding,
		core.CapabilityMultimodal,
		core.CapabilityVision,
		core.CapabilityAudio,
	}
	if modelID == core.ModelGoogleGeminiFlashImage {
		capabilities = append(capabilities, core.CapabilityImageGeneration)
	}

	// Check if streaming is supported
	if supportsGeminiStreaming(modelID) {
		capabilities = append(capabilities, core.CapabilityStreaming)
	}

	// Check if function calling is supported
	if supportsGeminiFunctionCalling(modelID) {
		capabilities = append(capabilities, core.CapabilityToolCalling)
	}

	llm := &GeminiLLM{
		apiKey:  apiKey,
		BaseLLM: core.NewBaseLLM("google", modelID, capabilities, endpoint),
	}
	_, _ = llm.ensureClient(ctx)
	return llm, nil
}

// GeminiProviderFactory creates GeminiLLM instances.
func GeminiProviderFactory(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	return NewGeminiLLMFromConfig(ctx, config, modelID)
}

// validGeminiModels is the list of supported Gemini model IDs.
var validGeminiModels = []core.ModelID{
	// Gemini 2.5 series (existing)
	core.ModelGoogleGeminiFlash,     // gemini-2.5-flash
	core.ModelGoogleGeminiPro,       // gemini-2.5-pro
	core.ModelGoogleGeminiFlashLite,  // gemini-2.5-flash-lite
	core.ModelGoogleGeminiFlashImage, // gemini-2.5-flash-image
	// Gemini 3 series (new)
	core.ModelGoogleGemini3ProPreview,   // gemini-3-pro-preview
	core.ModelGoogleGemini3FlashPreview, // gemini-3-flash-preview
	// Gemini 2.0 series (new)
	core.ModelGoogleGemini20Flash,     // gemini-2.0-flash
	core.ModelGoogleGemini20FlashLite, // gemini-2.0-flash-lite
}

// isValidGeminiModel checks if the model is a valid Gemini model.
func isValidGeminiModel(modelID core.ModelID) bool {
	return isValidModelInList(modelID, validGeminiModels)
}

// supportsGeminiStreaming checks if the model supports streaming.
// Currently all Gemini models support streaming, but this function is kept
// as a placeholder for future model-specific capability checks.
// The modelID parameter is unused but kept for API consistency.
func supportsGeminiStreaming(_ core.ModelID) bool {
	return true
}

// supportsGeminiFunctionCalling checks if the model supports function calling.
// Currently all Gemini models support function calling, but this function is kept
// as a placeholder for future model-specific capability checks.
// The modelID parameter is unused but kept for API consistency.
func supportsGeminiFunctionCalling(_ core.ModelID) bool {
	return true
}

const (
	geminiThoughtMetadataKey          = "gemini_thought"
	geminiThoughtSignatureMetadataKey = "gemini_thought_signature"
	geminiThoughtSignatureSkipValue   = "skip_thought_signature_validator"
)

func supportsGeminiThoughtSignatures(modelID core.ModelID) bool {
	model := strings.ToLower(strings.TrimSpace(string(modelID)))
	return strings.HasPrefix(model, "gemini-3")
}

// Gemini 3 thought signatures must be round-tripped when tool calling is active.
// We currently enable thoughts automatically for Gemini 3 family models so the
// caller receives the signatures needed for subsequent function-response turns.
func (g *GeminiLLM) buildGenerationConfig(opts *core.GenerateOptions) geminiGenerationConfig {
	cfg := geminiGenerationConfig{
		Temperature:        opts.Temperature,
		MaxOutputTokens:    opts.MaxTokens,
		TopP:               opts.TopP,
		ResponseModalities: normalizeGeminiResponseModalities(opts.ResponseModalities),
	}
	if supportsGeminiThoughtSignatures(core.ModelID(g.ModelID())) {
		cfg.ThinkingConfig = &geminiThinkingConfig{IncludeThoughts: true}
	}
	return cfg
}

// normalizeGeminiResponseModalities maps the dspy-go modality strings ("text",
// "image", "audio") to the Gemini API's uppercase form ("TEXT", "IMAGE",
// "AUDIO"). Unknown values are passed through as-is so callers can supply any
// future SDK constant.
func normalizeGeminiResponseModalities(modalities []string) []string {
	if len(modalities) == 0 {
		return nil
	}
	out := make([]string, 0, len(modalities))
	for _, m := range modalities {
		switch strings.ToLower(strings.TrimSpace(m)) {
		case "":
			continue
		case "text":
			out = append(out, "TEXT")
		case "image":
			out = append(out, "IMAGE")
		case "audio":
			out = append(out, "AUDIO")
		default:
			out = append(out, strings.ToUpper(m))
		}
	}
	return out
}

func geminiUsageToTokenInfo(metadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
	ThoughtsTokenCount   int `json:"thoughtsTokenCount"`
}) *core.TokenInfo {
	return &core.TokenInfo{
		PromptTokens:     metadata.PromptTokenCount,
		CompletionTokens: metadata.CandidatesTokenCount,
		TotalTokens:      metadata.TotalTokenCount,
	}
}

func geminiUsageMetadataMap(metadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
	ThoughtsTokenCount   int `json:"thoughtsTokenCount"`
}) map[string]any {
	result := map[string]any{}
	if metadata.ThoughtsTokenCount > 0 {
		result["thoughts_token_count"] = metadata.ThoughtsTokenCount
	}
	return result
}

func buildGeminiContentResponse(parts []geminiPart) (string, []core.ContentBlock, []core.ToolCall, map[string]any) {
	var (
		textParts     []string
		contentBlocks []core.ContentBlock
		toolCalls     []core.ToolCall
		thoughtBlocks []core.ContentBlock
	)

	for idx, part := range parts {
		if part.Text != "" {
			block := core.NewTextBlock(part.Text)
			if part.Thought {
				if block.Metadata == nil {
					block.Metadata = map[string]any{}
				}
				block.Metadata[geminiThoughtMetadataKey] = true
				if part.ThoughtSignature != "" {
					block.Metadata[geminiThoughtSignatureMetadataKey] = part.ThoughtSignature
				}
				thoughtBlocks = append(thoughtBlocks, block)
			} else {
				textParts = append(textParts, part.Text)
			}
			contentBlocks = append(contentBlocks, block)
		}

		if part.FunctionCall != nil {
			callID := strings.TrimSpace(part.FunctionCall.ID)
			if callID == "" {
				callID = fmt.Sprintf("gemini-call-%d", idx)
			}
			call := core.ToolCall{
				ID:        callID,
				Name:      part.FunctionCall.Name,
				Arguments: part.FunctionCall.Arguments,
			}
			if part.Thought || part.ThoughtSignature != "" {
				call.Metadata = map[string]any{}
				if part.Thought {
					call.Metadata[geminiThoughtMetadataKey] = true
				}
				if part.ThoughtSignature != "" {
					call.Metadata[geminiThoughtSignatureMetadataKey] = part.ThoughtSignature
				}
			}
			toolCalls = append(toolCalls, call)
		}

		// Image / audio outputs are returned as inlineData parts on the
		// candidate (e.g. gemini-2.5-flash-image returns generated images).
		// sdkPartToLegacy already base64-encodes the SDK Blob.Data into
		// InlineData.Data, so we decode it back to raw bytes for the caller.
		if part.InlineData != nil {
			data, err := base64.StdEncoding.DecodeString(part.InlineData.Data)
			if err != nil {
				// If the data isn't valid base64 (shouldn't happen given the
				// SDK round-trip) fall back to the raw string bytes so we
				// don't drop the payload entirely.
				data = []byte(part.InlineData.Data)
			}
			block := core.ContentBlock{
				Data:     data,
				MimeType: part.InlineData.MimeType,
			}
			if strings.HasPrefix(part.InlineData.MimeType, "audio/") {
				block.Type = core.FieldTypeAudio
			} else {
				block.Type = core.FieldTypeImage
			}
			contentBlocks = append(contentBlocks, block)
		}
	}

	metadata := map[string]any{}
	if len(thoughtBlocks) > 0 {
		metadata["thought_blocks"] = thoughtBlocks
	}
	if len(toolCalls) > 0 {
		metadata["tool_call_count"] = len(toolCalls)
	}

	return strings.Join(textParts, " "), contentBlocks, toolCalls, metadata
}

func (g *GeminiLLM) chatMessagesToGeminiContents(messages []core.ChatMessage) []geminiContent {
	contents := make([]geminiContent, 0, len(messages))
	for i := 0; i < len(messages); i++ {
		msg := messages[i]
		// Gemini expects tool results as function_response parts rather than a
		// standalone "tool" role. Consecutive tool results from the same native
		// turn are replayed as a single user content turn with multiple
		// functionResponse parts.
		if msg.Role == "tool" {
			content := geminiContent{
				Role:  geminiRoleForChatMessage(msg.Role),
				Parts: []geminiPart{},
			}
			for ; i < len(messages); i++ {
				toolMsg := messages[i]
				if toolMsg.Role != "tool" {
					i--
					break
				}
				if toolMsg.ToolResult == nil {
					continue
				}
				response := map[string]any{
					"content": contentBlocksToText(toolMsg.ToolResult.Content),
				}
				content.Parts = append(content.Parts, geminiPart{
					FunctionResponse: &geminiFunctionResponsePart{
						ID:       strings.TrimSpace(toolMsg.ToolResult.ToolCallID),
						Name:     toolMsg.ToolResult.Name,
						Response: response,
					},
				})
			}
			if len(content.Parts) > 0 {
				contents = append(contents, content)
			}
			continue
		}

		contentBlocks := msg.Content
		if len(msg.ToolCalls) > 0 {
			// Gemini validates function-call signatures for tool replay, but
			// replaying the full thought text significantly inflates the request
			// without being required for correctness.
			contentBlocks = filterGeminiAssistantToolReplayContent(contentBlocks)
		}

		content := geminiContent{
			Role:  geminiRoleForChatMessage(msg.Role),
			Parts: make([]geminiPart, 0, len(contentBlocks)+len(msg.ToolCalls)+1),
		}
		content.Parts = append(content.Parts, g.convertToGeminiParts(contentBlocks)...)

		for _, toolCall := range msg.ToolCalls {
			part := geminiPart{
				FunctionCall: &geminiFunctionCall{
					ID:        strings.TrimSpace(toolCall.ID),
					Name:      toolCall.Name,
					Arguments: toolCall.Arguments,
				},
			}
			if toolCall.Metadata != nil {
				if thought, ok := toolCall.Metadata[geminiThoughtMetadataKey].(bool); ok {
					part.Thought = thought
				}
				if signature, ok := toolCall.Metadata[geminiThoughtSignatureMetadataKey].(string); ok {
					part.ThoughtSignature = signature
				}
			}
			if part.ThoughtSignature == "" && supportsGeminiThoughtSignatures(core.ModelID(g.ModelID())) {
				part.ThoughtSignature = geminiThoughtSignatureSkipValue
			}
			content.Parts = append(content.Parts, part)
		}

		if len(content.Parts) == 0 {
			continue
		}
		contents = append(contents, content)
	}
	return contents
}

func filterGeminiAssistantToolReplayContent(blocks []core.ContentBlock) []core.ContentBlock {
	if len(blocks) == 0 {
		return nil
	}
	filtered := make([]core.ContentBlock, 0, len(blocks))
	for _, block := range blocks {
		if block.Metadata != nil {
			if thought, ok := block.Metadata[geminiThoughtMetadataKey].(bool); ok && thought {
				continue
			}
		}
		filtered = append(filtered, block)
	}
	return filtered
}

func geminiRoleForChatMessage(role string) string {
	switch role {
	case "assistant":
		return "model"
	default:
		return "user"
	}
}

// ensureClient lazily initializes the genai SDK client. The client is built
// from the apiKey and endpoint base URL on the GeminiLLM, which lets tests
// that construct GeminiLLM by hand (with an httptest.Server URL) keep working.
// The Gemini API backend is forced; Vertex AI is not supported by this layer.
func (g *GeminiLLM) ensureClient(ctx context.Context) (*genai.Client, error) {
	g.clientMu.Lock()
	defer g.clientMu.Unlock()
	if g.client != nil {
		return g.client, nil
	}
	if g.clientErr != nil {
		return nil, g.clientErr
	}

	cfg := &genai.ClientConfig{
		APIKey:  g.apiKey,
		Backend: genai.BackendGeminiAPI,
	}
	if endpoint := g.GetEndpointConfig(); endpoint != nil {
		if endpoint.BaseURL != "" {
			// genai's base URL is appended with "{apiVersion}/models/...";
			// it expects a trailing slash to demarcate the API root from the
			// resource path. Test servers (httptest.NewServer) hand back URLs
			// without a trailing slash, so we add one defensively.
			baseURL := strings.TrimRight(endpoint.BaseURL, "/") + "/"
			cfg.HTTPOptions.BaseURL = baseURL
		}
		if len(endpoint.Headers) > 0 {
			cfg.HTTPOptions.Headers = http.Header{}
			for k, v := range endpoint.Headers {
				// The SDK manages Content-Type itself; forwarding ours would
				// double up the header.
				if strings.EqualFold(k, "Content-Type") {
					continue
				}
				cfg.HTTPOptions.Headers.Set(k, v)
			}
		}
	}

	client, err := genai.NewClient(ctx, cfg)
	if err != nil {
		g.clientErr = err
		return nil, err
	}
	g.client = client
	return client, nil
}

// doGeminiRequest is the legacy entry point used by Generate, GenerateWithFunctions,
// GenerateWithTools, and GenerateWithJSON. It accepts a fully-constructed legacy
// geminiRequest, translates it to the SDK's typed call, and writes the result
// back into a legacy geminiResponse / geminiFunctionCallResponse so the calling
// methods see no behavioral change.
func (g *GeminiLLM) doGeminiRequest(ctx context.Context, reqBody any, dest any) error {
	legacyReq, err := coerceGeminiRequest(reqBody)
	if err != nil {
		return errors.WithFields(err, errors.Fields{"model": g.ModelID()})
	}

	client, err := g.ensureClient(ctx)
	if err != nil {
		return errors.WithFields(
			errors.New(errors.InvalidInput, fmt.Sprintf("failed to initialize Gemini client: %v", err)),
			errors.Fields{"model": g.ModelID()},
		)
	}

	contents := legacyContentsToSDK(legacyReq.Contents)
	config := legacyRequestToSDKConfig(legacyReq)

	sdkResp, err := client.Models.GenerateContent(ctx, g.ModelID(), contents, config)
	if err != nil {
		return wrapGeminiSDKError(err, g.ModelID())
	}

	legacyResp := sdkGenerateContentResponseToLegacy(sdkResp)
	return assignLegacyGeminiResponse(legacyResp, dest)
}

func coerceGeminiRequest(reqBody any) (geminiRequest, error) {
	switch v := reqBody.(type) {
	case geminiRequest:
		return v, nil
	case *geminiRequest:
		if v == nil {
			return geminiRequest{}, errors.New(errors.InvalidInput, "nil Gemini request body")
		}
		return *v, nil
	default:
		// Round-trip via JSON for any other shape so the calling code can be
		// liberal about request type. This branch is not exercised by the
		// existing call sites but keeps the helper future-proof.
		raw, err := json.Marshal(reqBody)
		if err != nil {
			return geminiRequest{}, errors.New(errors.InvalidInput, fmt.Sprintf("failed to marshal Gemini request body: %v", err))
		}
		var out geminiRequest
		if err := json.Unmarshal(raw, &out); err != nil {
			return geminiRequest{}, errors.New(errors.InvalidInput, fmt.Sprintf("failed to coerce Gemini request body: %v", err))
		}
		return out, nil
	}
}

// wrapGeminiSDKError converts a genai SDK error into the dspy-go error envelope
// the rest of the codebase (and tests) expect. The historical hand-rolled HTTP
// path produced messages like "API request failed with status code 503: ...";
// callers and tests assert against substrings of those messages, so the
// wrapper preserves them.
func wrapGeminiSDKError(err error, modelID string) error {
	if err == nil {
		return nil
	}
	var apiErr genai.APIError
	if stderrors.As(err, &apiErr) {
		message := strings.TrimSpace(apiErr.Message)
		if message == "" {
			message = strings.TrimSpace(apiErr.Status)
		}
		formatted := fmt.Sprintf("LLMGenerationFailed: API request failed with status code %d: %s", apiErr.Code, message)
		fields := errors.Fields{"model": modelID, "statusCode": apiErr.Code}
		if apiErr.Status != "" {
			fields["status"] = apiErr.Status
		}
		return errors.WithFields(
			errors.New(errors.LLMGenerationFailed, formatted),
			fields,
		)
	}
	// JSON decode failures inside the SDK look like "error unmarshalling response".
	low := strings.ToLower(err.Error())
	if strings.Contains(low, "unmarshalling response") || strings.Contains(low, "unmarshal") {
		return errors.WithFields(
			errors.New(errors.InvalidResponse, fmt.Sprintf("InvalidResponse: failed to unmarshal Gemini response: %v", err)),
			errors.Fields{"model": modelID},
		)
	}
	return errors.WithFields(
		errors.New(errors.LLMGenerationFailed, fmt.Sprintf("LLMGenerationFailed: failed to send Gemini request: %v", err)),
		errors.Fields{"model": modelID},
	)
}

// legacyContentsToSDK converts the dspy-go legacy geminiContent representation
// into the SDK's *genai.Content slice. It mirrors every field used by the
// existing helpers (text, inline data, file data, function calls, function
// responses, thoughts/signatures) so the round-trip is lossless.
func legacyContentsToSDK(contents []geminiContent) []*genai.Content {
	if len(contents) == 0 {
		return nil
	}
	out := make([]*genai.Content, 0, len(contents))
	for _, c := range contents {
		parts := make([]*genai.Part, 0, len(c.Parts))
		for _, p := range c.Parts {
			parts = append(parts, legacyPartToSDK(p))
		}
		out = append(out, &genai.Content{
			Parts: parts,
			Role:  c.Role,
		})
	}
	return out
}

func legacyPartToSDK(p geminiPart) *genai.Part {
	out := &genai.Part{
		Text:    p.Text,
		Thought: p.Thought,
	}
	if p.ThoughtSignature != "" {
		out.ThoughtSignature = []byte(p.ThoughtSignature)
	}
	if p.InlineData != nil {
		// Legacy inline data carries base64-encoded bytes; the SDK Blob wants
		// raw bytes. Decoding here keeps wire compatibility identical.
		data, err := base64.StdEncoding.DecodeString(p.InlineData.Data)
		if err != nil {
			data = []byte(p.InlineData.Data)
		}
		out.InlineData = &genai.Blob{
			MIMEType: p.InlineData.MimeType,
			Data:     data,
		}
	}
	if p.FileData != nil {
		out.FileData = &genai.FileData{
			MIMEType: p.FileData.MimeType,
			FileURI:  p.FileData.FileURI,
		}
	}
	if p.FunctionCall != nil {
		out.FunctionCall = &genai.FunctionCall{
			ID:   p.FunctionCall.ID,
			Name: p.FunctionCall.Name,
			Args: p.FunctionCall.Arguments,
		}
	}
	if p.FunctionResponse != nil {
		out.FunctionResponse = &genai.FunctionResponse{
			ID:       p.FunctionResponse.ID,
			Name:     p.FunctionResponse.Name,
			Response: p.FunctionResponse.Response,
		}
	}
	return out
}

// legacyRequestToSDKConfig folds the legacy geminiRequest's GenerationConfig,
// Tools, and ToolConfig into a single SDK GenerateContentConfig.
func legacyRequestToSDKConfig(req geminiRequest) *genai.GenerateContentConfig {
	cfg := &genai.GenerateContentConfig{}
	if req.GenerationConfig.Temperature != 0 {
		t := float32(req.GenerationConfig.Temperature)
		cfg.Temperature = &t
	}
	if req.GenerationConfig.MaxOutputTokens != 0 {
		cfg.MaxOutputTokens = int32(req.GenerationConfig.MaxOutputTokens)
	}
	if req.GenerationConfig.TopP != 0 {
		t := float32(req.GenerationConfig.TopP)
		cfg.TopP = &t
	}
	if req.GenerationConfig.ThinkingConfig != nil {
		cfg.ThinkingConfig = &genai.ThinkingConfig{
			IncludeThoughts: req.GenerationConfig.ThinkingConfig.IncludeThoughts,
		}
	}
	if len(req.GenerationConfig.ResponseModalities) > 0 {
		cfg.ResponseModalities = make([]string, len(req.GenerationConfig.ResponseModalities))
		copy(cfg.ResponseModalities, req.GenerationConfig.ResponseModalities)
	}
	if len(req.Tools) > 0 {
		cfg.Tools = make([]*genai.Tool, 0, len(req.Tools))
		for _, t := range req.Tools {
			decls := make([]*genai.FunctionDeclaration, 0, len(t.FunctionDeclarations))
			for _, d := range t.FunctionDeclarations {
				decl := &genai.FunctionDeclaration{
					Name:        d.Name,
					Description: d.Description,
				}
				if len(d.Parameters) > 0 {
					// Use the JSON-schema escape hatch so the legacy
					// map[string]interface{} schema flows through unchanged
					// without forcing us to translate to the SDK's typed Schema.
					decl.ParametersJsonSchema = d.Parameters
				}
				decls = append(decls, decl)
			}
			cfg.Tools = append(cfg.Tools, &genai.Tool{FunctionDeclarations: decls})
		}
	}
	if req.ToolConfig != nil && req.ToolConfig.FunctionCallingConfig != nil {
		cfg.ToolConfig = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode:                 genai.FunctionCallingConfigMode(req.ToolConfig.FunctionCallingConfig.Mode),
				AllowedFunctionNames: req.ToolConfig.FunctionCallingConfig.AllowedFunctionNames,
			},
		}
	}
	return cfg
}

// sdkGenerateContentResponseToLegacy folds an SDK GenerateContentResponse into
// a legacy geminiFunctionCallResponse, which is the superset of the two legacy
// response types (geminiResponse and geminiFunctionCallResponse share the same
// candidate/usage shape).
func sdkGenerateContentResponseToLegacy(resp *genai.GenerateContentResponse) geminiFunctionCallResponse {
	var out geminiFunctionCallResponse
	if resp == nil {
		return out
	}
	for _, cand := range resp.Candidates {
		var cs struct {
			Content struct {
				Parts []geminiPart `json:"parts"`
			} `json:"content"`
			FinishReason string `json:"finishReason,omitempty"`
		}
		if cand.Content != nil {
			parts := make([]geminiPart, 0, len(cand.Content.Parts))
			for _, p := range cand.Content.Parts {
				parts = append(parts, sdkPartToLegacy(p))
			}
			cs.Content.Parts = parts
		}
		cs.FinishReason = string(cand.FinishReason)
		out.Candidates = append(out.Candidates, cs)
	}
	if resp.UsageMetadata != nil {
		out.UsageMetadata.PromptTokenCount = int(resp.UsageMetadata.PromptTokenCount)
		out.UsageMetadata.CandidatesTokenCount = int(resp.UsageMetadata.CandidatesTokenCount)
		out.UsageMetadata.TotalTokenCount = int(resp.UsageMetadata.TotalTokenCount)
		out.UsageMetadata.ThoughtsTokenCount = int(resp.UsageMetadata.ThoughtsTokenCount)
	}
	if resp.PromptFeedback != nil {
		feedback := map[string]any{}
		if resp.PromptFeedback.BlockReason != "" {
			feedback["blockReason"] = string(resp.PromptFeedback.BlockReason)
		}
		if resp.PromptFeedback.BlockReasonMessage != "" {
			feedback["blockReasonMessage"] = resp.PromptFeedback.BlockReasonMessage
		}
		if len(feedback) > 0 {
			out.PromptFeedback = feedback
		}
	}
	return out
}

func sdkPartToLegacy(p *genai.Part) geminiPart {
	if p == nil {
		return geminiPart{}
	}
	out := geminiPart{
		Text:    p.Text,
		Thought: p.Thought,
	}
	if len(p.ThoughtSignature) > 0 {
		out.ThoughtSignature = string(p.ThoughtSignature)
	}
	if p.InlineData != nil {
		out.InlineData = &geminiInlineData{
			MimeType: p.InlineData.MIMEType,
			Data:     base64.StdEncoding.EncodeToString(p.InlineData.Data),
		}
	}
	if p.FileData != nil {
		out.FileData = &geminiFileData{
			MimeType: p.FileData.MIMEType,
			FileURI:  p.FileData.FileURI,
		}
	}
	if p.FunctionCall != nil {
		out.FunctionCall = &geminiFunctionCall{
			ID:        p.FunctionCall.ID,
			Name:      p.FunctionCall.Name,
			Arguments: p.FunctionCall.Args,
		}
	}
	if p.FunctionResponse != nil {
		out.FunctionResponse = &geminiFunctionResponsePart{
			ID:       p.FunctionResponse.ID,
			Name:     p.FunctionResponse.Name,
			Response: p.FunctionResponse.Response,
		}
	}
	return out
}

// assignLegacyGeminiResponse copies the merged legacy response into either of
// the two destination shapes used by callers.
func assignLegacyGeminiResponse(src geminiFunctionCallResponse, dest any) error {
	switch d := dest.(type) {
	case *geminiResponse:
		d.Candidates = src.Candidates
		d.UsageMetadata = src.UsageMetadata
		return nil
	case *geminiFunctionCallResponse:
		*d = src
		return nil
	default:
		// Fall back to JSON round-trip for any unexpected destination shape.
		raw, err := json.Marshal(src)
		if err != nil {
			return err
		}
		return json.Unmarshal(raw, dest)
	}
}

func contentBlocksToText(blocks []core.ContentBlock) string {
	parts := make([]string, 0, len(blocks))
	for _, block := range blocks {
		if block.Type == core.FieldTypeText && block.Text != "" {
			parts = append(parts, block.Text)
		}
	}
	return strings.Join(parts, "\n")
}

func toolSchemasToDeclarations(tools []map[string]interface{}) ([]geminiFunctionDeclaration, error) {
	functionDeclarations := make([]geminiFunctionDeclaration, 0, len(tools))
	for _, tool := range tools {
		name, ok := tool["name"].(string)
		if !ok {
			return nil, errors.WithFields(
				errors.New(errors.InvalidInput, "tool schema missing 'name' field"),
				errors.Fields{"tool": tool},
			)
		}
		description, _ := tool["description"].(string)
		parameters, ok := tool["parameters"].(map[string]interface{})
		if !ok {
			return nil, errors.WithFields(
				errors.New(errors.InvalidInput, "tool schema missing 'parameters' field"),
				errors.Fields{"tool": tool},
			)
		}

		functionDeclarations = append(functionDeclarations, geminiFunctionDeclaration{
			Name:        name,
			Description: description,
			Parameters:  parameters,
		})
	}
	return functionDeclarations, nil
}

// Generate implements the core.LLM interface.
func (g *GeminiLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	reqBody := geminiRequest{
		Contents: []geminiContent{
			{
				Parts: []geminiPart{
					{Text: prompt},
				},
			},
		},
		GenerationConfig: g.buildGenerationConfig(opts),
	}
	var geminiResp geminiResponse
	if err := g.doGeminiRequest(ctx, reqBody, &geminiResp); err != nil {
		return nil, errors.WithFields(
			err,
			errors.Fields{
				"prompt": prompt,
				"model":  g.ModelID(),
			})
	}

	if len(geminiResp.Candidates) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "InvalidResponse: no candidates in response"),
			errors.Fields{
				"model": g.ModelID(),
			})
	}

	if len(geminiResp.Candidates[0].Content.Parts) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "InvalidResponse: no parts in response candidate"),
			errors.Fields{
				"model": g.ModelID(),
			})
	}

	content, contentBlocks, _, responseMetadata := buildGeminiContentResponse(geminiResp.Candidates[0].Content.Parts)
	usage := geminiUsageToTokenInfo(geminiResp.UsageMetadata)
	metadata := geminiUsageMetadataMap(geminiResp.UsageMetadata)
	for key, value := range responseMetadata {
		metadata[key] = value
	}

	return &core.LLMResponse{
		Content:       content,
		ContentBlocks: contentBlocks,
		Usage:         usage,
		Metadata:      metadata,
	}, nil
}

// GenerateWithJSON implements the core.LLM interface.
func (g *GeminiLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	response, err := g.Generate(ctx, prompt, options...)
	if err != nil {
		return nil, err
	}

	return utils.ParseJSONResponse(response.Content)
}

// Implement the GenerateWithFunctions method for GeminiLLM.
func (g *GeminiLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
	logger := logging.GetLogger()
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	functionDeclarations, err := toolSchemasToDeclarations(functions)
	if err != nil {
		return nil, err
	}

	// Create the request body with functions
	reqBody := geminiRequest{
		Contents: []geminiContent{
			{
				Parts: []geminiPart{
					{Text: prompt},
				},
				Role: "user",
			},
		},
		Tools: []geminiTool{
			{
				FunctionDeclarations: functionDeclarations,
			},
		},
		ToolConfig:       requiredGeminiToolConfig(functionDeclarations),
		GenerationConfig: g.buildGenerationConfig(opts),
	}
	requestJSON, _ := json.MarshalIndent(reqBody, "", "  ")
	logger.Debug(ctx, "Function call request JSON: %s", string(requestJSON))

	// Parse the response
	var geminiResp geminiFunctionCallResponse
	if err := g.doGeminiRequest(ctx, reqBody, &geminiResp); err != nil {
		return nil, errors.WithFields(
			err,
			errors.Fields{
				"prompt": prompt,
				"model":  g.ModelID(),
			})
	}

	if len(geminiResp.Candidates) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "no candidates in response"),
			errors.Fields{
				"model": g.ModelID(),
			})
	}

	// Extract usage information
	usage := geminiUsageToTokenInfo(geminiResp.UsageMetadata)

	// Process the response to extract function call if present
	result := make(map[string]interface{})

	// Check if there are any parts in the response
	if len(geminiResp.Candidates[0].Content.Parts) > 0 {
		textContent, contentBlocks, toolCalls, responseMetadata := buildGeminiContentResponse(geminiResp.Candidates[0].Content.Parts)

		// Add text content if available
		if textContent != "" {
			result["content"] = textContent
		}

		if len(contentBlocks) > 0 {
			result["content_blocks"] = contentBlocks
		}

		if len(toolCalls) > 0 {
			result["tool_calls"] = toolCalls
			call := toolCalls[0]
			result["function_call"] = map[string]interface{}{
				"id":        call.ID,
				"name":      call.Name,
				"arguments": call.Arguments,
			}
			if len(call.Metadata) > 0 {
				result["function_call"].(map[string]interface{})["metadata"] = call.Metadata
			}
		}

		for key, value := range responseMetadata {
			result[key] = value
		}
	}

	// If no content or function call was found, add a default message and stamp
	// provider diagnostics so callers can distinguish an actually empty Gemini
	// tool-call response from a normal plain-text reply.
	if len(result) == 0 {
		result["content"] = "No content or function call received from model"
		result["provider_diagnostic"] = geminiEmptyToolResponseDiagnostic(
			"functions",
			geminiResp.Candidates[0].FinishReason,
			len(geminiResp.Candidates),
			len(geminiResp.Candidates[0].Content.Parts),
			geminiResp.PromptFeedback,
		)
	}

	// Add token usage information
	result["_usage"] = usage

	return result, nil
}

// GenerateWithTools implements native multi-turn tool calling for Gemini.
func (g *GeminiLLM) GenerateWithTools(ctx context.Context, messages []core.ChatMessage, tools []map[string]any, options ...core.GenerateOption) (map[string]interface{}, error) {
	logger := logging.GetLogger()
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	functionDeclarations, err := toolSchemasToDeclarations(tools)
	if err != nil {
		return nil, err
	}

	reqBody := geminiRequest{
		Contents:         g.chatMessagesToGeminiContents(messages),
		GenerationConfig: g.buildGenerationConfig(opts),
	}
	if len(functionDeclarations) > 0 {
		reqBody.Tools = []geminiTool{{FunctionDeclarations: functionDeclarations}}
		reqBody.ToolConfig = requiredGeminiToolConfig(functionDeclarations)
	}

	requestJSON, _ := json.MarshalIndent(reqBody, "", "  ")
	logger.Debug(ctx, "Gemini tool request JSON: %s", string(requestJSON))

	var geminiResp geminiFunctionCallResponse
	if err := g.doGeminiRequest(ctx, reqBody, &geminiResp); err != nil {
		return nil, errors.WithFields(
			err,
			errors.Fields{"model": g.ModelID()},
		)
	}
	if len(geminiResp.Candidates) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "no candidates in Gemini tool response"),
			errors.Fields{"model": g.ModelID()},
		)
	}

	textContent, contentBlocks, toolCalls, responseMetadata := buildGeminiContentResponse(geminiResp.Candidates[0].Content.Parts)
	result := map[string]any{
		"_usage": geminiUsageToTokenInfo(geminiResp.UsageMetadata),
	}
	if textContent != "" {
		result["content"] = textContent
	}
	if len(contentBlocks) > 0 {
		result["content_blocks"] = contentBlocks
	}
	if len(toolCalls) > 0 {
		result["tool_calls"] = toolCalls
		first := toolCalls[0]
		functionCall := map[string]any{
			"id":        first.ID,
			"name":      first.Name,
			"arguments": first.Arguments,
		}
		if len(first.Metadata) > 0 {
			functionCall["metadata"] = first.Metadata
		}
		result["function_call"] = functionCall
	}
	for key, value := range geminiUsageMetadataMap(geminiResp.UsageMetadata) {
		result[key] = value
	}
	for key, value := range responseMetadata {
		result[key] = value
	}
	if _, hasContent := result["content"]; !hasContent && len(toolCalls) == 0 {
		result["content"] = "No content or function call received from model"
		result["provider_diagnostic"] = geminiEmptyToolResponseDiagnostic(
			"tools",
			geminiResp.Candidates[0].FinishReason,
			len(geminiResp.Candidates),
			len(geminiResp.Candidates[0].Content.Parts),
			geminiResp.PromptFeedback,
		)
	}

	return result, nil
}

func requiredGeminiToolConfig(functions []geminiFunctionDeclaration) *geminiToolConfig {
	if len(functions) == 0 {
		return nil
	}

	names := make([]string, 0, len(functions))
	for _, function := range functions {
		if strings.TrimSpace(function.Name) != "" {
			names = append(names, function.Name)
		}
	}
	if len(names) == 0 {
		return nil
	}

	return &geminiToolConfig{
		FunctionCallingConfig: &geminiFunctionCallingConfig{
			Mode:                 "ANY",
			AllowedFunctionNames: names,
		},
	}
}

func geminiEmptyToolResponseDiagnostic(mode, finishReason string, candidateCount, partCount int, promptFeedback map[string]any) map[string]any {
	diagnostic := map[string]any{
		"provider":        "google",
		"provider_mode":   mode,
		"reason":          "empty_content_and_function_call",
		"candidate_count": candidateCount,
		"part_count":      partCount,
	}
	if strings.TrimSpace(finishReason) != "" {
		diagnostic["finish_reason"] = finishReason
	}
	if len(promptFeedback) > 0 {
		diagnostic["prompt_feedback"] = promptFeedback
	}
	return diagnostic
}

// CreateEmbedding implements the embedding generation for a single input.
func (g *GeminiLLM) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	// Apply options
	opts := core.NewEmbeddingOptions()
	for _, opt := range options {
		opt(opts)
	}
	if opts.Model == "" {
		opts.Model = "text-embedding-004"
	} else if !isValidGeminiEmbeddingModel(opts.Model) {
		return nil, errors.New(errors.InvalidInput, fmt.Sprintf("invalid Gemini embedding model: %s", opts.Model))
	}

	client, err := g.ensureClient(ctx)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to initialize Gemini client"),
			errors.Fields{"model": opts.Model},
		)
	}

	cfg := buildGeminiEmbedConfig(opts)
	contents := []*genai.Content{
		genai.NewContentFromText(input, genai.RoleUser),
	}

	resp, err := client.Models.EmbedContent(ctx, opts.Model, contents, cfg)
	if err != nil {
		return nil, errors.WithFields(
			wrapGeminiEmbeddingSDKError(err, opts.Model),
			errors.Fields{"input_length": len(input)},
		)
	}

	if len(resp.Embeddings) == 0 || len(resp.Embeddings[0].Values) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "embedding values missing in response"),
			errors.Fields{"model": opts.Model},
		)
	}

	embedding := resp.Embeddings[0]
	result := &core.EmbeddingResult{
		Vector:     embedding.Values,
		TokenCount: embeddingTokenCount(embedding),
		Metadata: map[string]interface{}{
			"model":            opts.Model,
			"prompt_tokens":    embeddingTokenCount(embedding),
			"truncated_tokens": embeddingTruncatedTokens(embedding),
			"embedding_tokens": embeddingTokenCount(embedding),
		},
	}
	return result, nil
}

// buildGeminiEmbedConfig translates dspy-go's EmbeddingOptions into the SDK's
// EmbedContentConfig. Only the task_type parameter is currently meaningful for
// the legacy callers; the rest of opts.Params is ignored because the SDK does
// not have a typed equivalent on the Gemini backend.
func buildGeminiEmbedConfig(opts *core.EmbeddingOptions) *genai.EmbedContentConfig {
	if opts == nil {
		return nil
	}
	cfg := &genai.EmbedContentConfig{}
	hasField := false
	if taskType, ok := opts.Params["task_type"].(string); ok && taskType != "" {
		cfg.TaskType = taskType
		hasField = true
	}
	if title, ok := opts.Params["title"].(string); ok && title != "" {
		cfg.Title = title
		hasField = true
	}
	if !hasField {
		return nil
	}
	return cfg
}

func embeddingTokenCount(e *genai.ContentEmbedding) int {
	if e == nil || e.Statistics == nil {
		return 0
	}
	return int(e.Statistics.TokenCount)
}

func embeddingTruncatedTokens(e *genai.ContentEmbedding) int {
	if e == nil || e.Statistics == nil || !e.Statistics.Truncated {
		return 0
	}
	return int(e.Statistics.TokenCount)
}

func wrapGeminiEmbeddingSDKError(err error, model string) error {
	if err == nil {
		return nil
	}
	var apiErr genai.APIError
	if stderrors.As(err, &apiErr) {
		message := strings.TrimSpace(apiErr.Message)
		if message == "" {
			message = strings.TrimSpace(apiErr.Status)
		}
		return errors.WithFields(
			errors.New(errors.LLMGenerationFailed, fmt.Sprintf("API request failed with status code %d: %s", apiErr.Code, message)),
			errors.Fields{"model": model, "statusCode": apiErr.Code},
		)
	}
	low := strings.ToLower(err.Error())
	if strings.Contains(low, "unmarshal") {
		return errors.WithFields(
			errors.Wrap(err, errors.InvalidResponse, "failed to unmarshal response"),
			errors.Fields{"model": model},
		)
	}
	return errors.WithFields(
		errors.Wrap(err, errors.LLMGenerationFailed, "failed to send request"),
		errors.Fields{"model": model},
	)
}

// CreateEmbeddings implements batch embedding generation.
func (g *GeminiLLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	opts := core.NewEmbeddingOptions()
	for _, opt := range options {
		opt(opts)
	}

	if opts.BatchSize <= 0 {
		opts.BatchSize = 32
	}

	const model = "text-embedding-004"

	client, err := g.ensureClient(ctx)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to initialize Gemini client"),
			errors.Fields{"model": model},
		)
	}

	cfg := buildGeminiEmbedConfig(opts)

	var allResults []core.EmbeddingResult
	var firstError error
	errorIndex := -1

	for i := 0; i < len(inputs); i += opts.BatchSize {
		end := i + opts.BatchSize
		if end > len(inputs) {
			end = len(inputs)
		}
		batch := inputs[i:end]

		contents := make([]*genai.Content, 0, len(batch))
		for _, in := range batch {
			contents = append(contents, genai.NewContentFromText(in, genai.RoleUser))
		}

		resp, embedErr := client.Models.EmbedContent(ctx, model, contents, cfg)
		if embedErr != nil {
			if firstError == nil {
				firstError = wrapGeminiEmbeddingSDKError(embedErr, model)
				errorIndex = i
			}
			continue
		}

		for j, embedding := range resp.Embeddings {
			result := core.EmbeddingResult{
				Vector:     embedding.Values,
				TokenCount: embeddingTokenCount(embedding),
				Metadata: map[string]interface{}{
					"model":            model,
					"prompt_tokens":    embeddingTokenCount(embedding),
					"truncated_tokens": embeddingTruncatedTokens(embedding),
					"embedding_tokens": embeddingTokenCount(embedding),
					"batch_index":      i + j,
				},
			}
			allResults = append(allResults, result)
		}
	}

	if firstError != nil && len(allResults) == 0 {
		return nil, firstError
	}

	return &core.BatchEmbeddingResult{
		Embeddings: allResults,
		Error:      firstError,
		ErrorIndex: errorIndex,
	}, nil
}

// streamRequest handles the common streaming logic for both StreamGenerate and StreamGenerateWithContent.
// The body is the legacy geminiRequest; we route the call through the genai SDK
// streaming iterator and feed each chunk into the dspy-go StreamResponse channel.
func (g *GeminiLLM) streamRequest(ctx context.Context, reqBody interface{}) (*core.StreamResponse, error) {
	legacyReq, err := coerceGeminiRequest(reqBody)
	if err != nil {
		return nil, errors.WithFields(err, errors.Fields{"model": g.ModelID()})
	}

	client, err := g.ensureClient(ctx)
	if err != nil {
		return nil, errors.WithFields(
			errors.New(errors.InvalidInput, fmt.Sprintf("failed to initialize Gemini client: %v", err)),
			errors.Fields{"model": g.ModelID()})
	}

	contents := legacyContentsToSDK(legacyReq.Contents)
	config := legacyRequestToSDKConfig(legacyReq)

	chunkChan := make(chan core.StreamChunk)
	streamCtx, cancelStream := context.WithCancel(ctx)

	var channelClosed sync.Once
	safeCloseChannel := func() {
		channelClosed.Do(func() {
			close(chunkChan)
		})
	}

	response := &core.StreamResponse{
		ChunkChannel: chunkChan,
		Cancel: func() {
			cancelStream()
		},
	}

	go func() {
		defer safeCloseChannel()

		seq := client.Models.GenerateContentStream(streamCtx, g.ModelID(), contents, config)
		seq(func(resp *genai.GenerateContentResponse, iterErr error) bool {
			if streamCtx.Err() != nil {
				return false
			}
			if iterErr != nil {
				// Skip the SDK's "[DONE]" parse error which arises because the
				// real Gemini API does not send a [DONE] sentinel but some
				// test fixtures do. Any other error is real and is forwarded.
				if isGeminiDoneSentinelError(iterErr) {
					return false
				}
				select {
				case chunkChan <- core.StreamChunk{
					Error: wrapGeminiSDKError(iterErr, g.ModelID()),
				}:
				case <-streamCtx.Done():
				}
				return false
			}
			if resp == nil || len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
				return true
			}
			for _, part := range resp.Candidates[0].Content.Parts {
				if part == nil || part.Text == "" {
					continue
				}
				select {
				case chunkChan <- core.StreamChunk{Content: part.Text}:
				case <-streamCtx.Done():
					return false
				}
			}
			return true
		})
	}()

	return response, nil
}

// isGeminiDoneSentinelError matches the error genai.iterateResponseStream
// emits when it tries to JSON-decode a `data: [DONE]` line. The real Gemini
// API never emits this, but historical test fixtures still do, and treating it
// as graceful end-of-stream keeps those fixtures green.
func isGeminiDoneSentinelError(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "[DONE]") || strings.Contains(msg, "invalid character '['")
}

// StreamGenerate for Gemini.
func (g *GeminiLLM) StreamGenerate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	reqBody := geminiRequest{
		Contents: []geminiContent{
			{
				Parts: []geminiPart{
					{Text: prompt},
				},
			},
		},
		GenerationConfig: geminiGenerationConfig{
			Temperature:     opts.Temperature,
			MaxOutputTokens: opts.MaxTokens,
			TopP:            opts.TopP,
		},
	}

	return g.streamRequest(ctx, reqBody)
}

func isValidGeminiEmbeddingModel(s string) bool {
	validModels := []string{
		"gemini-embedding-exp-03-07",
		"text-embedding-004",
		"gemini-embedding-004",
		"embedding-001",
		"embedding-latest",
		"embedding-gecko",
		"embedding-gecko-001",
		"text-embedding-gecko-001",
	}

	for _, model := range validModels {
		if s == model {
			return true
		}
	}

	return false
}

func constructRequestURL(endpoint *core.EndpointConfig, apiKey string) string {
	// Remove any trailing slashes from base URL and leading slashes from path
	baseURL := strings.TrimRight(endpoint.BaseURL, "/")
	path := strings.TrimLeft(endpoint.Path, "/")

	// Join them with a single slash
	fullEndpoint := fmt.Sprintf("%s/%s", baseURL, path)

	// Add the API key as query parameter
	return fmt.Sprintf("%s?key=%s", fullEndpoint, apiKey)
}

// GenerateWithContent implements multimodal content generation for Gemini.
func (g *GeminiLLM) GenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.LLMResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	// Convert ContentBlocks to Gemini's format
	geminiParts := g.convertToGeminiParts(content)

	reqBody := geminiRequest{
		Contents: []geminiContent{
			{
				Parts: geminiParts,
			},
		},
		GenerationConfig: g.buildGenerationConfig(opts),
	}

	var geminiResp geminiResponse
	if err := g.doGeminiRequest(ctx, reqBody, &geminiResp); err != nil {
		return nil, errors.WithFields(
			err,
			errors.Fields{
				"content_blocks": len(content),
				"model":          g.ModelID(),
			})
	}

	if len(geminiResp.Candidates) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "no candidates in response"),
			errors.Fields{
				"model": g.ModelID(),
			})
	}

	if len(geminiResp.Candidates[0].Content.Parts) == 0 {
		return nil, errors.WithFields(
			errors.New(errors.InvalidResponse, "no parts in response candidate"),
			errors.Fields{
				"model": g.ModelID(),
			})
	}

	contentText, contentBlocks, _, responseMetadata := buildGeminiContentResponse(geminiResp.Candidates[0].Content.Parts)
	usage := geminiUsageToTokenInfo(geminiResp.UsageMetadata)
	metadata := geminiUsageMetadataMap(geminiResp.UsageMetadata)
	for key, value := range responseMetadata {
		metadata[key] = value
	}

	return &core.LLMResponse{
		Content:       contentText,
		ContentBlocks: contentBlocks,
		Usage:         usage,
		Metadata:      metadata,
	}, nil
}

// StreamGenerateWithContent implements multimodal streaming for Gemini.
func (g *GeminiLLM) StreamGenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.StreamResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	// Convert ContentBlocks to Gemini's format
	geminiParts := g.convertToGeminiParts(content)

	reqBody := geminiRequest{
		Contents: []geminiContent{
			{
				Parts: geminiParts,
			},
		},
		GenerationConfig: g.buildGenerationConfig(opts),
	}

	return g.streamRequest(ctx, reqBody)
}

// convertToGeminiParts converts ContentBlocks to Gemini's format.
func (g *GeminiLLM) convertToGeminiParts(blocks []core.ContentBlock) []geminiPart {
	var parts []geminiPart

	for _, block := range blocks {
		switch block.Type {
		case core.FieldTypeText:
			part := geminiPart{
				Text: block.Text,
			}
			if block.Metadata != nil {
				if thought, ok := block.Metadata[geminiThoughtMetadataKey].(bool); ok {
					part.Thought = thought
				}
				if signature, ok := block.Metadata[geminiThoughtSignatureMetadataKey].(string); ok {
					part.ThoughtSignature = signature
				}
			}
			parts = append(parts, part)
		case core.FieldTypeImage:
			parts = append(parts, geminiPart{
				InlineData: &geminiInlineData{
					MimeType: block.MimeType,
					Data:     base64.StdEncoding.EncodeToString(block.Data),
				},
			})
		case core.FieldTypeAudio:
			parts = append(parts, geminiPart{
				InlineData: &geminiInlineData{
					MimeType: block.MimeType,
					Data:     base64.StdEncoding.EncodeToString(block.Data),
				},
			})
		default:
			// Fallback to text
			parts = append(parts, geminiPart{
				Text: fmt.Sprintf("[Unsupported content type: %s]", block.Type),
			})
		}
	}

	return parts
}
