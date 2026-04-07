package llms

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/shared"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
)

// OpenAILLM implements the core.LLM interface for OpenAI's models.
type OpenAILLM struct {
	*core.BaseLLM
	apiKey string
	client openaisdk.Client
}

// OpenAIOption is a functional option for configuring OpenAI provider.
type OpenAIOption func(*OpenAIConfig)

// OpenAIConfig holds configuration for OpenAI provider.
type OpenAIConfig struct {
	baseURL    string
	path       string
	apiKey     string
	headers    map[string]string
	timeout    time.Duration
	httpClient *http.Client
}

// isOpenAIOAuthToken checks if the token is an OAuth token from ChatGPT Plus/Pro subscription.
func isOpenAIOAuthToken(token string) bool {
	// OpenAI OAuth tokens from auth.openai.com typically start with specific prefixes
	return strings.HasPrefix(token, "sess-") || strings.HasPrefix(token, "oat-")
}

// NewOpenAILLM creates a new OpenAILLM instance with functional options.
func NewOpenAILLM(modelID core.ModelID, opts ...OpenAIOption) (*OpenAILLM, error) {
	config := &OpenAIConfig{
		baseURL: "https://api.openai.com", // default
		path:    "/v1/chat/completions",
		timeout: 60 * time.Second,
		headers: make(map[string]string),
	}

	for _, opt := range opts {
		opt(config)
	}

	// Environment variable fallback for API key
	// Priority: OPENAI_OAUTH_TOKEN > config.apiKey > OPENAI_API_KEY
	if config.apiKey == "" {
		config.apiKey = os.Getenv("OPENAI_OAUTH_TOKEN")
	}
	if config.apiKey == "" {
		config.apiKey = os.Getenv("OPENAI_API_KEY")
	}

	// API key validation - required for official OpenAI API endpoint
	if config.apiKey == "" && config.baseURL == "https://api.openai.com" {
		return nil, errors.WithFields(
			errors.New(errors.InvalidInput, "OpenAI API key or OAuth token required"),
			errors.Fields{"env_vars": "OPENAI_API_KEY or OPENAI_OAUTH_TOKEN"})
	}

	// Build endpoint configuration. We continue to expose the endpoint config
	// (including Authorization header) so callers and tests can introspect it,
	// even though the underlying HTTP transport is now driven by the SDK.
	endpointCfg := &core.EndpointConfig{
		BaseURL:    config.baseURL,
		Path:       config.path,
		Headers:    config.headers,
		TimeoutSec: int(config.timeout.Seconds()),
	}

	// Set authorization header and OAuth-specific headers if needed
	if config.apiKey != "" {
		endpointCfg.Headers["Authorization"] = "Bearer " + config.apiKey

		// Add OAuth-specific headers for ChatGPT Plus/Pro subscriptions
		if isOpenAIOAuthToken(config.apiKey) {
			endpointCfg.Headers["openai-beta"] = "assistants=v2"
			endpointCfg.Headers["x-openai-client"] = "dspy-go"
		}
	}
	endpointCfg.Headers["Content-Type"] = "application/json"

	capabilities := []core.Capability{
		core.CapabilityCompletion,
		core.CapabilityChat,
		core.CapabilityJSON,
		core.CapabilityStreaming,
		core.CapabilityEmbedding,
		core.CapabilityToolCalling,
	}

	baseLLM := core.NewBaseLLM("openai", modelID, capabilities, endpointCfg)

	client := buildOpenAISDKClient(config)

	return &OpenAILLM{
		BaseLLM: baseLLM,
		apiKey:  config.apiKey,
		client:  client,
	}, nil
}

// buildOpenAISDKClient constructs an openai-go SDK client from the user-supplied
// configuration. It translates baseURL+path into the SDK's base URL convention
// (which always ends in a trailing slash and has the route path appended) and
// forwards any custom headers and HTTP client.
func buildOpenAISDKClient(config *OpenAIConfig) openaisdk.Client {
	sdkBase := computeOpenAISDKBaseURL(config.baseURL, config.path)

	opts := []option.RequestOption{
		option.WithBaseURL(sdkBase),
		// Disable the SDK's automatic retries; the dspy-go layer manages its
		// own retry semantics (notably the invalid_request_error JSON-parse
		// fallback) and tests assert on exact request counts.
		option.WithMaxRetries(0),
	}
	if config.apiKey != "" {
		opts = append(opts, option.WithAPIKey(config.apiKey))
	}
	if config.httpClient != nil {
		opts = append(opts, option.WithHTTPClient(config.httpClient))
	}
	for k, v := range config.headers {
		// Skip headers the SDK manages itself.
		if strings.EqualFold(k, "Authorization") || strings.EqualFold(k, "Content-Type") {
			continue
		}
		opts = append(opts, option.WithHeader(k, v))
	}
	return openaisdk.NewClient(opts...)
}

// computeOpenAISDKBaseURL derives the SDK base URL from the legacy baseURL+path
// pair. The SDK appends route paths like "chat/completions" and "embeddings"
// directly to the base URL, so we strip a trailing chat/completions segment
// from the legacy path and ensure a trailing slash.
func computeOpenAISDKBaseURL(baseURL, path string) string {
	full := strings.TrimRight(baseURL, "/")
	if path != "" {
		if !strings.HasPrefix(path, "/") {
			full += "/"
		}
		full += path
	}
	full = strings.TrimSuffix(full, "/chat/completions")
	full = strings.TrimSuffix(full, "chat/completions")
	if !strings.HasSuffix(full, "/") {
		full += "/"
	}
	return full
}

// Option functions for OpenAI configuration

// WithAPIKey sets the API key.
func WithAPIKey(apiKey string) OpenAIOption {
	return func(c *OpenAIConfig) { c.apiKey = apiKey }
}

// WithOpenAIBaseURL sets the base URL.
func WithOpenAIBaseURL(baseURL string) OpenAIOption {
	return func(c *OpenAIConfig) { c.baseURL = baseURL }
}

// WithOpenAIPath sets the endpoint path.
func WithOpenAIPath(path string) OpenAIOption {
	return func(c *OpenAIConfig) { c.path = path }
}

// WithOpenAITimeout sets the request timeout.
func WithOpenAITimeout(timeout time.Duration) OpenAIOption {
	return func(c *OpenAIConfig) { c.timeout = timeout }
}

// WithHeader sets a custom header.
func WithHeader(key, value string) OpenAIOption {
	return func(c *OpenAIConfig) {
		if c.headers == nil {
			c.headers = make(map[string]string)
		}
		c.headers[key] = value
	}
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(client *http.Client) OpenAIOption {
	return func(c *OpenAIConfig) { c.httpClient = client }
}

// Convenience constructor for standard OpenAI.
func NewOpenAI(modelID core.ModelID, apiKey string) (*OpenAILLM, error) {
	if apiKey == "" {
		return nil, errors.WithFields(
			errors.New(errors.InvalidInput, "OpenAI API key is required"),
			errors.Fields{"env_var": "OPENAI_API_KEY"})
	}
	return NewOpenAILLM(modelID, WithAPIKey(apiKey))
}

// NewOpenAILLMFromConfig creates a new OpenAILLM instance from configuration.
func NewOpenAILLMFromConfig(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (*OpenAILLM, error) {
	opts := []OpenAIOption{}

	// Priority: OPENAI_OAUTH_TOKEN > config.APIKey > OPENAI_API_KEY
	apiKey := os.Getenv("OPENAI_OAUTH_TOKEN")
	if apiKey == "" {
		apiKey = config.APIKey
	}
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}
	if apiKey != "" {
		opts = append(opts, WithAPIKey(apiKey))
	}

	// Set base URL from config.BaseURL or config.Endpoint.BaseURL (endpoint takes priority)
	baseURL := config.BaseURL
	if config.Endpoint != nil && config.Endpoint.BaseURL != "" {
		baseURL = config.Endpoint.BaseURL
	}
	if baseURL != "" {
		opts = append(opts, WithOpenAIBaseURL(baseURL))
	}

	// Validate model ID only for the official OpenAI API endpoint to allow custom models with compatible APIs.
	// An empty baseURL defaults to the official OpenAI API.
	if (baseURL == "" || baseURL == "https://api.openai.com") && !isValidOpenAIModel(modelID) {
		return nil, errors.WithFields(
			errors.New(errors.InvalidInput, "unsupported model for official OpenAI API"),
			errors.Fields{"model": modelID})
	}

	if config.Endpoint != nil && config.Endpoint.Path != "" {
		opts = append(opts, WithOpenAIPath(config.Endpoint.Path))
	}

	if config.Endpoint != nil && config.Endpoint.TimeoutSec > 0 {
		opts = append(opts, WithOpenAITimeout(time.Duration(config.Endpoint.TimeoutSec)*time.Second))
	}

	if config.Endpoint != nil {
		for key, value := range config.Endpoint.Headers {
			opts = append(opts, WithHeader(key, value))
		}
	}

	return NewOpenAILLM(modelID, opts...)
}

// OpenAIProviderFactory creates OpenAILLM instances.
func OpenAIProviderFactory(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	return NewOpenAILLMFromConfig(ctx, config, modelID)
}

// validOpenAIModels is the list of supported OpenAI model IDs.
var validOpenAIModels = []core.ModelID{
	// GPT-4 series
	core.ModelOpenAIGPT4,
	core.ModelOpenAIGPT4Turbo,
	core.ModelOpenAIGPT35Turbo,
	core.ModelOpenAIGPT4o,
	core.ModelOpenAIGPT4oMini,
	// GPT-4.1 series
	core.ModelOpenAIGPT41,
	core.ModelOpenAIGPT41Mini,
	core.ModelOpenAIGPT41Nano,
	// o1 reasoning models
	core.ModelOpenAIO1,
	core.ModelOpenAIO1Pro,
	core.ModelOpenAIO1Mini,
	// o3 reasoning models
	core.ModelOpenAIO3,
	core.ModelOpenAIO3Mini,
	// GPT-5 series
	core.ModelOpenAIGPT5,
	core.ModelOpenAIGPT5Mini,
	core.ModelOpenAIGPT5Nano,
	// GPT-5.2 series (instant, thinking, pro, codex)
	core.ModelOpenAIGPT52,
	core.ModelOpenAIGPT52Instant,
	core.ModelOpenAIGPT52Thinking,
	core.ModelOpenAIGPT52ThinkHigh,
	core.ModelOpenAIGPT52Pro,
	core.ModelOpenAIGPT52Codex,
}

// isValidOpenAIModel checks if the model is a valid OpenAI model.
func isValidOpenAIModel(modelID core.ModelID) bool {
	return isValidModelInList(modelID, validOpenAIModels)
}

// Generate implements the core.LLM interface.
func (o *OpenAILLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
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

	response, err := o.chatCompletion(ctx, params)
	if err != nil {
		return nil, err
	}

	if len(response.Choices) == 0 {
		return nil, errors.New(errors.InvalidResponse, "no choices returned from OpenAI API")
	}

	usage := &core.TokenInfo{
		PromptTokens:     int(response.Usage.PromptTokens),
		CompletionTokens: int(response.Usage.CompletionTokens),
		TotalTokens:      int(response.Usage.TotalTokens),
	}

	return &core.LLMResponse{
		Content: response.Choices[0].Message.Content,
		Usage:   usage,
		Metadata: map[string]interface{}{
			"finish_reason": response.Choices[0].FinishReason,
			"id":            response.ID,
			"model":         response.Model,
		},
	}, nil
}

// GenerateWithJSON implements the core.LLM interface.
func (o *OpenAILLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	params := openaisdk.ChatCompletionNewParams{
		Model: shared.ChatModel(o.ModelID()),
		Messages: []openaisdk.ChatCompletionMessageParamUnion{
			openaisdk.UserMessage(prompt),
		},
		ResponseFormat: openaisdk.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONObject: &shared.ResponseFormatJSONObjectParam{},
		},
	}
	applyOpenAIGenerateOptions(&params, o.ModelID(), opts)

	response, err := o.chatCompletion(ctx, params)
	if err != nil {
		return nil, err
	}

	if len(response.Choices) == 0 {
		return nil, errors.New(errors.InvalidResponse, "no choices returned from OpenAI API")
	}

	return utils.ParseJSONResponse(response.Choices[0].Message.Content)
}

// GenerateWithFunctions implements the core.LLM interface.
func (o *OpenAILLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
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

	response, err := o.chatCompletion(ctx, params)
	if err != nil {
		return nil, err
	}

	if len(response.Choices) == 0 {
		return nil, errors.New(errors.InvalidResponse, "no choices returned from OpenAI API")
	}

	return buildOpenAIToolResultFromSDK(response.Choices[0], response.Usage, "functions")
}

// GenerateWithTools implements native multi-turn tool calling for OpenAI.
func (o *OpenAILLM) GenerateWithTools(ctx context.Context, messages []core.ChatMessage, tools []map[string]any, options ...core.GenerateOption) (map[string]interface{}, error) {
	if len(tools) == 0 {
		return nil, errors.New(errors.InvalidInput, "at least one tool schema is required")
	}

	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	openAIMessages, err := convertCoreChatMessagesToOpenAISDK(messages)
	if err != nil {
		return nil, err
	}

	openAITools, err := convertOpenAIAnyToolSchemasToSDK(tools)
	if err != nil {
		return nil, err
	}

	params := openaisdk.ChatCompletionNewParams{
		Model:    shared.ChatModel(o.ModelID()),
		Messages: openAIMessages,
		Tools:    openAITools,
		ToolChoice: openaisdk.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: param.NewOpt("auto"),
		},
	}
	applyOpenAIGenerateOptions(&params, o.ModelID(), opts)

	response, err := o.chatCompletion(ctx, params)
	if err != nil {
		return nil, err
	}

	if len(response.Choices) == 0 {
		return nil, errors.New(errors.InvalidResponse, "no choices returned from OpenAI API")
	}

	return buildOpenAIToolResultFromSDK(response.Choices[0], response.Usage, "tools")
}

func convertOpenAIFunctionSchemasToSDK(functions []map[string]interface{}) ([]openaisdk.ChatCompletionToolUnionParam, error) {
	tools := make([]openaisdk.ChatCompletionToolUnionParam, 0, len(functions))

	for _, function := range functions {
		name, ok := function["name"].(string)
		if !ok || strings.TrimSpace(name) == "" {
			return nil, errors.New(errors.InvalidInput, "function schema missing non-empty 'name' field")
		}

		description, _ := function["description"].(string)

		parameters := shared.FunctionParameters{"type": "object"}
		if rawParameters, hasParameters := function["parameters"]; hasParameters && rawParameters != nil {
			paramMap, ok := rawParameters.(map[string]interface{})
			if !ok {
				return nil, errors.WithFields(
					errors.New(errors.InvalidInput, "function schema 'parameters' must be an object"),
					errors.Fields{"function_name": name},
				)
			}

			parameters = shared.FunctionParameters(cloneMap(paramMap))
			if _, hasType := parameters["type"]; !hasType {
				parameters["type"] = "object"
			}
		}

		functionDef := shared.FunctionDefinitionParam{
			Name:       name,
			Parameters: parameters,
		}
		if description != "" {
			functionDef.Description = param.NewOpt(description)
		}

		tools = append(tools, openaisdk.ChatCompletionToolUnionParam{
			OfFunction: &openaisdk.ChatCompletionFunctionToolParam{
				Function: functionDef,
			},
		})
	}

	return tools, nil
}

func convertOpenAIAnyToolSchemasToSDK(functions []map[string]any) ([]openaisdk.ChatCompletionToolUnionParam, error) {
	converted := make([]map[string]interface{}, 0, len(functions))
	for _, function := range functions {
		converted = append(converted, anyMapToInterfaceMap(function))
	}
	return convertOpenAIFunctionSchemasToSDK(converted)
}

func parseOpenAIFunctionArguments(raw string) (map[string]interface{}, error) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return map[string]interface{}{}, nil
	}

	var parsed interface{}
	if err := json.Unmarshal([]byte(trimmed), &parsed); err != nil {
		return nil, err
	}

	arguments, ok := parsed.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("function arguments must decode to a JSON object")
	}

	return arguments, nil
}

func cloneMap(in map[string]interface{}) map[string]interface{} {
	out := make(map[string]interface{}, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func anyMapToInterfaceMap(in map[string]any) map[string]interface{} {
	out := make(map[string]interface{}, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

// convertCoreChatMessagesToOpenAISDK translates dspy-go's core.ChatMessage values
// (which can carry assistant tool calls and tool results) into the openai-go
// SDK's ChatCompletionMessageParamUnion. The pendingToolCallIDs bookkeeping is
// the same as the previous hand-rolled converter: a tool result whose ToolCallID
// is unset is implicitly bound to the immediately preceding assistant turn IFF
// that turn produced exactly one tool call. Multi-call ambiguity is rejected.
func convertCoreChatMessagesToOpenAISDK(messages []core.ChatMessage) ([]openaisdk.ChatCompletionMessageParamUnion, error) {
	out := make([]openaisdk.ChatCompletionMessageParamUnion, 0, len(messages))
	pendingToolCallIDs := make([]string, 0, 1)

	for messageIndex, message := range messages {
		role := strings.TrimSpace(message.Role)
		content := flattenCoreChatMessageContent(message.Content)

		switch role {
		case "system":
			out = append(out, openaisdk.SystemMessage(content))
			pendingToolCallIDs = pendingToolCallIDs[:0]
		case "developer":
			out = append(out, openaisdk.DeveloperMessage(content))
			pendingToolCallIDs = pendingToolCallIDs[:0]
		case "user":
			out = append(out, openaisdk.UserMessage(content))
			pendingToolCallIDs = pendingToolCallIDs[:0]
		case "assistant":
			asst := openaisdk.ChatCompletionAssistantMessageParam{}
			if content != "" {
				asst.Content.OfString = param.NewOpt(content)
			}
			if len(message.ToolCalls) > 0 {
				pendingToolCallIDs = pendingToolCallIDs[:0]
				asst.ToolCalls = make([]openaisdk.ChatCompletionMessageToolCallUnionParam, 0, len(message.ToolCalls))
				for toolIndex, toolCall := range message.ToolCalls {
					argumentsJSON, err := json.Marshal(toolCall.Arguments)
					if err != nil {
						return nil, errors.WithFields(
							errors.Wrap(err, errors.InvalidInput, "failed to encode assistant tool call arguments"),
							errors.Fields{"tool_name": toolCall.Name},
						)
					}

					toolCallID := strings.TrimSpace(toolCall.ID)
					if toolCallID == "" {
						toolCallID = fmt.Sprintf("call_%d_%d", messageIndex, toolIndex)
					}
					pendingToolCallIDs = append(pendingToolCallIDs, toolCallID)

					asst.ToolCalls = append(asst.ToolCalls, openaisdk.ChatCompletionMessageToolCallUnionParam{
						OfFunction: &openaisdk.ChatCompletionMessageFunctionToolCallParam{
							ID: toolCallID,
							Function: openaisdk.ChatCompletionMessageFunctionToolCallFunctionParam{
								Name:      toolCall.Name,
								Arguments: string(argumentsJSON),
							},
						},
					})
				}
			} else {
				pendingToolCallIDs = pendingToolCallIDs[:0]
			}
			out = append(out, openaisdk.ChatCompletionMessageParamUnion{OfAssistant: &asst})
		case "tool":
			toolMsg := openaisdk.ChatCompletionToolMessageParam{}
			if message.ToolResult != nil {
				toolContent := flattenCoreChatMessageContent(message.ToolResult.Content)
				toolMsg.Content.OfString = param.NewOpt(toolContent)

				toolCallID := strings.TrimSpace(message.ToolResult.ToolCallID)
				if toolCallID == "" {
					switch len(pendingToolCallIDs) {
					case 0:
					case 1:
						toolCallID = pendingToolCallIDs[0]
					default:
						return nil, errors.Wrap(
							fmt.Errorf("assistant turn has %d pending tool calls", len(pendingToolCallIDs)),
							errors.InvalidInput,
							"tool result is missing tool call id",
						)
					}
				}
				toolMsg.ToolCallID = toolCallID
				if toolCallID != "" {
					pendingToolCallIDs = removePendingToolCallID(pendingToolCallIDs, toolCallID)
				}
			}
			out = append(out, openaisdk.ChatCompletionMessageParamUnion{OfTool: &toolMsg})
		default:
			// Unknown roles fall back to user content to keep prior behavior.
			out = append(out, openaisdk.UserMessage(content))
			pendingToolCallIDs = pendingToolCallIDs[:0]
		}
	}

	return out, nil
}

func removePendingToolCallID(ids []string, target string) []string {
	for i, id := range ids {
		if id != target {
			continue
		}
		return append(ids[:i], ids[i+1:]...)
	}
	return ids
}

func flattenCoreChatMessageContent(blocks []core.ContentBlock) string {
	if len(blocks) == 0 {
		return ""
	}

	parts := make([]string, 0, len(blocks))
	for _, block := range blocks {
		text := strings.TrimSpace(block.String())
		if text == "" {
			continue
		}
		parts = append(parts, text)
	}
	return strings.Join(parts, "\n")
}

// buildOpenAIToolResultFromSDK shapes an SDK ChatCompletionChoice into the
// historical map[string]interface{} contract: a "tool_calls" slice of
// core.ToolCall, plus a "function_call" convenience map for legacy callers,
// plus token usage on "_usage".
func buildOpenAIToolResultFromSDK(choice openaisdk.ChatCompletionChoice, usage openaisdk.CompletionUsage, mode string) (map[string]interface{}, error) {
	result := map[string]interface{}{}

	if content := strings.TrimSpace(choice.Message.Content); content != "" {
		result["content"] = content
	}

	if len(choice.Message.ToolCalls) > 0 {
		toolCalls := make([]core.ToolCall, 0, len(choice.Message.ToolCalls))
		for _, call := range choice.Message.ToolCalls {
			fn := call.AsFunction()
			args, err := parseOpenAIFunctionArguments(fn.Function.Arguments)
			if err != nil {
				return nil, errors.WithFields(
					errors.Wrap(err, errors.InvalidResponse, "failed to parse tool call arguments"),
					errors.Fields{"tool_name": fn.Function.Name},
				)
			}
			toolCalls = append(toolCalls, core.ToolCall{
				ID:        fn.ID,
				Name:      fn.Function.Name,
				Arguments: args,
			})
		}
		result["tool_calls"] = toolCalls
		result["function_call"] = map[string]interface{}{
			"name":      toolCalls[0].Name,
			"arguments": toolCalls[0].Arguments,
		}
	} else if choice.Message.FunctionCall.Name != "" {
		args, err := parseOpenAIFunctionArguments(choice.Message.FunctionCall.Arguments)
		if err != nil {
			return nil, errors.WithFields(
				errors.Wrap(err, errors.InvalidResponse, "failed to parse legacy function call arguments"),
				errors.Fields{"tool_name": choice.Message.FunctionCall.Name},
			)
		}
		result["function_call"] = map[string]interface{}{
			"name":      choice.Message.FunctionCall.Name,
			"arguments": args,
		}
	}

	if len(result) == 0 {
		result["content"] = "No content or function call received from model"
		result["provider_diagnostic"] = map[string]any{
			"provider":      "openai",
			"provider_mode": mode,
			"reason":        "empty_content_and_function_call",
			"finish_reason": choice.FinishReason,
		}
	}

	result["_usage"] = &core.TokenInfo{
		PromptTokens:     int(usage.PromptTokens),
		CompletionTokens: int(usage.CompletionTokens),
		TotalTokens:      int(usage.TotalTokens),
	}

	return result, nil
}

// CreateEmbedding implements the core.LLM interface.
func (o *OpenAILLM) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	opts := core.NewEmbeddingOptions()
	for _, opt := range options {
		opt(opts)
	}

	model := "text-embedding-3-small"
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

	response, err := o.client.Embeddings.New(ctx, params)
	if err != nil {
		return nil, mapOpenAISDKError(err)
	}

	if len(response.Data) == 0 {
		return nil, errors.New(errors.InvalidResponse, "no embeddings returned from OpenAI API")
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
		},
	}, nil
}

// CreateEmbeddings implements the core.LLM interface.
func (o *OpenAILLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	opts := core.NewEmbeddingOptions()
	for _, opt := range options {
		opt(opts)
	}

	model := "text-embedding-3-small"
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

	response, err := o.client.Embeddings.New(ctx, params)
	if err != nil {
		return &core.BatchEmbeddingResult{Error: mapOpenAISDKError(err)}, nil
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
				"model": response.Model,
				"index": int(data.Index),
			},
		}
	}

	return &core.BatchEmbeddingResult{
		Embeddings: results,
	}, nil
}

// StreamGenerate implements the core.LLM interface.
func (o *OpenAILLM) StreamGenerate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
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
	streamCtx, cancelFunc := context.WithCancel(ctx)

	go func() {
		defer close(chunkChan)
		defer cancelFunc()

		stream := o.client.Chat.Completions.NewStreaming(streamCtx, params)
		defer func() {
			_ = stream.Close()
		}()

		pumpOpenAIChatCompletionStream(streamCtx, stream, chunkChan)
	}()

	return &core.StreamResponse{
		ChunkChannel: chunkChan,
		Cancel:       cancelFunc,
	}, nil
}

// pumpOpenAIChatCompletionStream forwards chunks from the SDK ssestream to the
// dspy-go channel contract. It mirrors the prior bufio-based reader: only
// deltas with non-empty content emit a Content chunk, finish_reason emits a
// Done chunk, and an SDK-level error becomes a wrapped LLMGenerationFailed
// error. Shared between OpenAI and the Ollama OpenAI-compatible mode.
func pumpOpenAIChatCompletionStream(ctx context.Context, stream *ssestream.Stream[openaisdk.ChatCompletionChunk], chunkChan chan<- core.StreamChunk) {
	for stream.Next() {
		select {
		case <-ctx.Done():
			return
		default:
		}

		chunk := stream.Current()
		if len(chunk.Choices) == 0 {
			continue
		}
		choice := chunk.Choices[0]
		if choice.Delta.Content != "" {
			select {
			case chunkChan <- core.StreamChunk{Content: choice.Delta.Content}:
			case <-ctx.Done():
				return
			}
		}
		if choice.FinishReason != "" {
			select {
			case chunkChan <- core.StreamChunk{Done: true}:
			case <-ctx.Done():
			}
			return
		}
	}

	if err := stream.Err(); err != nil {
		select {
		case chunkChan <- core.StreamChunk{
			Error: errors.Wrap(err, errors.LLMGenerationFailed, "streaming request failed"),
		}:
		case <-ctx.Done():
		}
		return
	}
	// SDK closed cleanly without an explicit finish_reason — emit Done.
	select {
	case chunkChan <- core.StreamChunk{Done: true}:
	case <-ctx.Done():
	}
}

// chatCompletion sends a chat completion request, applying the dspy-go retry
// quirk: a 400 invalid_request_error whose message indicates the API failed to
// parse the JSON body is retried exactly once with `Connection: close` to
// force a fresh TCP connection. This works around a sticky upstream bug where
// kept-alive connections occasionally see a request body get truncated.
func (o *OpenAILLM) chatCompletion(ctx context.Context, params openaisdk.ChatCompletionNewParams) (*openaisdk.ChatCompletion, error) {
	return o.chatCompletionWithRetry(ctx, params, true)
}

func (o *OpenAILLM) chatCompletionWithRetry(ctx context.Context, params openaisdk.ChatCompletionNewParams, allowRetry bool) (*openaisdk.ChatCompletion, error) {
	logger := logging.GetLogger()

	var reqOpts []option.RequestOption
	if !allowRetry {
		reqOpts = append(reqOpts, option.WithHeader("Connection", "close"))
	}

	response, err := o.client.Chat.Completions.New(ctx, params, reqOpts...)
	if err == nil {
		return response, nil
	}

	var apiErr *openaisdk.Error
	if !stderrors.As(err, &apiErr) {
		return nil, errors.Wrap(err, errors.LLMGenerationFailed, "request failed")
	}

	fields := errors.Fields{
		"type": apiErr.Type,
		"code": apiErr.Code,
	}
	if apiErr.Type == "invalid_request_error" {
		jsonData, _ := json.Marshal(params)
		summary := summarizeOpenAIParams(&params, jsonData)
		dumpPath, dumpErr := writeOpenAIInvalidRequestDebug(&params, jsonData, apiErr.StatusCode, []byte(apiErr.RawJSON()))
		logger.Error(
			ctx,
			"OpenAI invalid request: model=%s body_bytes=%d messages=%d tools=%d tool_choice=%s max_tokens=%v max_completion_tokens=%v dump=%s response=%s",
			summary.Model,
			summary.BodyBytes,
			summary.MessageCount,
			summary.ToolCount,
			summary.ToolChoice,
			summary.MaxTokens,
			summary.MaxCompletionTokens,
			dumpPath,
			strings.TrimSpace(apiErr.Message),
		)
		fields["request_body_bytes"] = summary.BodyBytes
		fields["request_body_sha256"] = summary.BodySHA256
		fields["message_count"] = summary.MessageCount
		fields["tool_count"] = summary.ToolCount
		fields["tool_choice"] = summary.ToolChoice
		if dumpPath != "" {
			fields["debug_dump"] = dumpPath
		}
		if dumpErr != nil {
			fields["debug_dump_error"] = dumpErr.Error()
		}
		if allowRetry && shouldRetryOpenAIInvalidJSONMessage(apiErr.Message) {
			logger.Warn(
				ctx,
				"Retrying OpenAI request after transient invalid_request_error with a fresh connection: model=%s dump=%s",
				summary.Model,
				dumpPath,
			)
			return o.chatCompletionWithRetry(ctx, params, false)
		}
	}

	return nil, errors.WithFields(
		errors.New(errors.LLMGenerationFailed, apiErr.Message),
		fields)
}

// mapOpenAISDKError converts a non-retried SDK error into a dspy-go errors.Error.
// Used for the embeddings code path which has no retry quirk.
func mapOpenAISDKError(err error) error {
	if err == nil {
		return nil
	}
	var apiErr *openaisdk.Error
	if stderrors.As(err, &apiErr) {
		return errors.WithFields(
			errors.New(errors.LLMGenerationFailed, apiErr.Message),
			errors.Fields{"type": apiErr.Type, "code": apiErr.Code})
	}
	return errors.Wrap(err, errors.LLMGenerationFailed, "request failed")
}

func shouldRetryOpenAIInvalidJSONMessage(message string) bool {
	normalized := strings.ToLower(strings.TrimSpace(message))
	return strings.Contains(normalized, "parse the json body")
}

// applyOpenAIGenerateOptions translates core.GenerateOptions into the SDK's
// ChatCompletionNewParams, preserving the historical model-aware behaviors:
//   - GPT-5.x models route MaxTokens to MaxCompletionTokens.
//   - GPT-5.x models do not support a custom Temperature; it is silently dropped.
func applyOpenAIGenerateOptions(params *openaisdk.ChatCompletionNewParams, model string, opts *core.GenerateOptions) {
	if opts == nil {
		return
	}

	if opts.MaxTokens > 0 {
		if openAIModelUsesMaxCompletionTokens(model) {
			params.MaxCompletionTokens = param.NewOpt(int64(opts.MaxTokens))
		} else {
			params.MaxTokens = param.NewOpt(int64(opts.MaxTokens))
		}
	}
	if opts.Temperature >= 0 && openAIModelSupportsCustomTemperature(model) {
		params.Temperature = param.NewOpt(opts.Temperature)
	}
	if opts.TopP > 0 {
		params.TopP = param.NewOpt(opts.TopP)
	}
	if opts.FrequencyPenalty != 0 {
		params.FrequencyPenalty = param.NewOpt(opts.FrequencyPenalty)
	}
	if opts.PresencePenalty != 0 {
		params.PresencePenalty = param.NewOpt(opts.PresencePenalty)
	}
	if len(opts.Stop) > 0 {
		params.Stop = openaisdk.ChatCompletionNewParamsStopUnion{
			OfStringArray: append([]string(nil), opts.Stop...),
		}
	}
}

func openAIModelUsesMaxCompletionTokens(model string) bool {
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(model)), "gpt-5")
}

func openAIModelSupportsCustomTemperature(model string) bool {
	return !openAIModelUsesMaxCompletionTokens(model)
}

// --- diagnostic dump helpers (used by the invalid_request_error path) ---

type openAIMessageDebugSummary struct {
	Role            string `json:"role"`
	ContentBytes    int    `json:"content_bytes"`
	ToolCalls       int    `json:"tool_calls,omitempty"`
	HasFunctionCall bool   `json:"has_function_call,omitempty"`
	HasToolCallID   bool   `json:"has_tool_call_id,omitempty"`
}

type openAIRequestDebugSummary struct {
	Model               string                      `json:"model"`
	BodyBytes           int                         `json:"body_bytes"`
	BodySHA256          string                      `json:"body_sha256"`
	MessageCount        int                         `json:"message_count"`
	Messages            []openAIMessageDebugSummary `json:"messages"`
	ToolCount           int                         `json:"tool_count,omitempty"`
	ToolNames           []string                    `json:"tool_names,omitempty"`
	ToolChoice          string                      `json:"tool_choice,omitempty"`
	MaxTokens           *int                        `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int                        `json:"max_completion_tokens,omitempty"`
	Temperature         *float64                    `json:"temperature,omitempty"`
}

type openAIInvalidRequestDebugDump struct {
	Summary        openAIRequestDebugSummary `json:"summary"`
	RequestBody    json.RawMessage           `json:"request_body"`
	ResponseStatus int                       `json:"response_status"`
	ResponseBody   string                    `json:"response_body"`
}

// summarizeOpenAIParams reads back fields from the marshaled JSON request body
// because the SDK's param types are unions whose Go-level "set" state is not
// trivially introspectable. Reading the JSON gives a single source of truth
// that matches the wire format we actually sent.
func summarizeOpenAIParams(params *openaisdk.ChatCompletionNewParams, jsonData []byte) openAIRequestDebugSummary {
	summary := openAIRequestDebugSummary{
		BodyBytes: len(jsonData),
	}
	if len(jsonData) > 0 {
		hash := sha256.Sum256(jsonData)
		summary.BodySHA256 = hex.EncodeToString(hash[:])
	}
	if params == nil {
		return summary
	}

	var decoded struct {
		Model               string            `json:"model"`
		Messages            []json.RawMessage `json:"messages"`
		Tools               []struct {
			Function struct {
				Name string `json:"name"`
			} `json:"function"`
		} `json:"tools"`
		ToolChoice          json.RawMessage `json:"tool_choice"`
		MaxTokens           *int            `json:"max_tokens"`
		MaxCompletionTokens *int            `json:"max_completion_tokens"`
		Temperature         *float64        `json:"temperature"`
	}
	if len(jsonData) > 0 {
		_ = json.Unmarshal(jsonData, &decoded)
	}

	summary.Model = decoded.Model
	summary.MessageCount = len(decoded.Messages)
	summary.ToolCount = len(decoded.Tools)
	summary.MaxTokens = decoded.MaxTokens
	summary.MaxCompletionTokens = decoded.MaxCompletionTokens
	summary.Temperature = decoded.Temperature

	if len(decoded.ToolChoice) > 0 {
		var asString string
		if err := json.Unmarshal(decoded.ToolChoice, &asString); err == nil {
			summary.ToolChoice = asString
		} else {
			summary.ToolChoice = strings.TrimSpace(string(decoded.ToolChoice))
		}
	}

	if len(decoded.Messages) > 0 {
		summary.Messages = make([]openAIMessageDebugSummary, 0, len(decoded.Messages))
		for _, raw := range decoded.Messages {
			var msg struct {
				Role         string            `json:"role"`
				Content      json.RawMessage   `json:"content"`
				ToolCalls    []json.RawMessage `json:"tool_calls"`
				FunctionCall json.RawMessage   `json:"function_call"`
				ToolCallID   string            `json:"tool_call_id"`
			}
			_ = json.Unmarshal(raw, &msg)

			contentBytes := 0
			if len(msg.Content) > 0 {
				var asString string
				if err := json.Unmarshal(msg.Content, &asString); err == nil {
					contentBytes = len(asString)
				} else {
					contentBytes = len(msg.Content)
				}
			}

			summary.Messages = append(summary.Messages, openAIMessageDebugSummary{
				Role:            msg.Role,
				ContentBytes:    contentBytes,
				ToolCalls:       len(msg.ToolCalls),
				HasFunctionCall: len(msg.FunctionCall) > 0 && string(msg.FunctionCall) != "null",
				HasToolCallID:   strings.TrimSpace(msg.ToolCallID) != "",
			})
		}
	}

	if len(decoded.Tools) > 0 {
		summary.ToolNames = make([]string, 0, len(decoded.Tools))
		for _, tool := range decoded.Tools {
			name := strings.TrimSpace(tool.Function.Name)
			if name == "" {
				name = "<unnamed>"
			}
			summary.ToolNames = append(summary.ToolNames, name)
		}
	}

	return summary
}

func writeOpenAIInvalidRequestDebug(params *openaisdk.ChatCompletionNewParams, jsonData []byte, responseStatus int, responseBody []byte) (string, error) {
	file, err := os.CreateTemp(os.TempDir(), "dspy-openai-invalid-request-*.json")
	if err != nil {
		return "", err
	}
	defer file.Close()

	dump := openAIInvalidRequestDebugDump{
		Summary:        summarizeOpenAIParams(params, jsonData),
		RequestBody:    json.RawMessage(append([]byte(nil), jsonData...)),
		ResponseStatus: responseStatus,
		ResponseBody:   string(responseBody),
	}

	encoded, err := json.MarshalIndent(dump, "", "  ")
	if err != nil {
		return "", err
	}
	if _, err := file.Write(encoded); err != nil {
		return "", err
	}
	return file.Name(), nil
}
