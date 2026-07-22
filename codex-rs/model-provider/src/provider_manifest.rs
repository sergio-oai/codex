use std::collections::HashSet;

use codex_models_manager::bundled_models_response;
use codex_models_manager::model_info::BASE_INSTRUCTIONS;
use codex_protocol::config_types::ReasoningSummary;
use codex_protocol::openai_models::ConfigShellToolType;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ModelServiceTier;
use codex_protocol::openai_models::ModelVisibility;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::openai_models::ReasoningEffortPreset;
use codex_protocol::openai_models::TruncationPolicyConfig;
use codex_protocol::openai_models::WebSearchToolType;
use codex_protocol::openai_models::default_input_modalities;
use serde::Deserialize;

const SUPPORTED_SCHEMA_VERSION: u32 = 1;
// Keep manifest-provided token limits in the range that downstream code can
// safely multiply while deriving compaction thresholds.
const MAX_SAFE_MODEL_TOKEN_LIMIT: i64 = i64::MAX / 9;

/// Safe provider-owned metadata used to build an authoritative model catalog.
///
/// The manifest deliberately cannot provide instructions, tools, headers, or
/// URLs. Those remain local Codex decisions; the provider controls only the
/// model names and capabilities that its API actually serves.
#[derive(Debug, Deserialize)]
struct ProviderManifest {
    schema_version: u32,
    models: Vec<ProviderManifestModel>,
}

#[derive(Debug, Deserialize)]
struct ProviderManifestModel {
    id: String,
    display_name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    context_window: Option<i64>,
    #[serde(default)]
    max_input_tokens: Option<i64>,
    #[serde(default)]
    default_reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    supported_reasoning_efforts: Vec<ReasoningEffort>,
    #[serde(default)]
    supports_personality: bool,
    #[serde(default)]
    service_tiers: Vec<ProviderManifestServiceTier>,
}

#[derive(Debug, Deserialize)]
struct ProviderManifestServiceTier {
    id: String,
    name: String,
    description: String,
}

pub(crate) fn parse_provider_manifest(body: &[u8]) -> Result<Vec<ModelInfo>, String> {
    let manifest: ProviderManifest = serde_json::from_slice(body)
        .map_err(|err| format!("failed to decode provider manifest: {err}"))?;
    if manifest.schema_version != SUPPORTED_SCHEMA_VERSION {
        return Err(format!(
            "unsupported provider manifest schema_version {}; expected {SUPPORTED_SCHEMA_VERSION}",
            manifest.schema_version
        ));
    }
    if manifest.models.is_empty() {
        return Err("provider manifest must contain at least one model".to_string());
    }

    let bundled_models = bundled_models_response()
        .map_err(|err| format!("failed to load bundled model metadata: {err}"))?
        .models;
    let mut seen_ids = HashSet::new();
    manifest
        .models
        .into_iter()
        .enumerate()
        .map(|(priority, manifest_model)| {
            to_model_info(manifest_model, priority, &bundled_models, &mut seen_ids)
        })
        .collect()
}

fn to_model_info(
    manifest_model: ProviderManifestModel,
    priority: usize,
    bundled_models: &[ModelInfo],
    seen_ids: &mut HashSet<String>,
) -> Result<ModelInfo, String> {
    if manifest_model.id.is_empty() || manifest_model.id.trim() != manifest_model.id {
        return Err("provider manifest model id must be non-empty and trimmed".to_string());
    }
    if !seen_ids.insert(manifest_model.id.clone()) {
        return Err(format!(
            "provider manifest contains duplicate model id {}",
            manifest_model.id
        ));
    }

    let context_window = positive_limit("context_window", manifest_model.context_window)?;
    let max_input_tokens = positive_limit("max_input_tokens", manifest_model.max_input_tokens)?;
    if manifest_model
        .default_reasoning_effort
        .as_ref()
        .is_some_and(|default| !manifest_model.supported_reasoning_efforts.contains(default))
    {
        return Err(format!(
            "provider manifest model {} has an unsupported default_reasoning_effort",
            manifest_model.id
        ));
    }
    // Codex uses this value to decide when to compact. Use the most
    // conservative known limit so the client compacts before it can send a
    // prompt that the provider rejects.
    let resolved_context_window = match (context_window, max_input_tokens) {
        (Some(context_window), Some(max_input_tokens)) => {
            Some(context_window.min(max_input_tokens))
        }
        (Some(context_window), None) => Some(context_window),
        (None, Some(max_input_tokens)) => Some(max_input_tokens),
        (None, None) => None,
    };
    let priority = i32::try_from(priority)
        .map_err(|_| "provider manifest contains too many models".to_string())?;

    let bundled_model = bundled_models
        .iter()
        .find(|model| model.slug == manifest_model.id);
    let mut model = safe_provider_manifest_model_info(&manifest_model.id, bundled_model);
    model.slug = manifest_model.id;
    model.display_name = if manifest_model.display_name.trim().is_empty() {
        model.slug.clone()
    } else {
        manifest_model.display_name
    };
    model.description =
        (!manifest_model.description.is_empty()).then_some(manifest_model.description);
    model.default_reasoning_level = manifest_model.default_reasoning_effort;
    model.supported_reasoning_levels = manifest_model
        .supported_reasoning_efforts
        .into_iter()
        .map(|effort| reasoning_preset(&model, effort))
        .collect();
    model.visibility = ModelVisibility::List;
    model.supported_in_api = true;
    model.priority = priority;
    model.additional_speed_tiers.clear();
    model.service_tiers = manifest_model
        .service_tiers
        .into_iter()
        .map(to_service_tier)
        .collect::<Result<Vec<_>, _>>()?;
    model.default_service_tier = None;
    // A provider-owned catalog should not inherit OpenAI-only lifecycle UX or
    // route a helper request to a model that the provider did not advertise.
    model.availability_nux = None;
    model.upgrade = None;
    model.auto_review_model_override = None;
    model.context_window = resolved_context_window;
    model.max_context_window = resolved_context_window;
    model.auto_compact_token_limit = None;
    model.comp_hash = None;
    // Personality prompts stay local to the binary. A manifest can disable a
    // locally supported personality, but it cannot provide or enable prompts.
    if !manifest_model.supports_personality {
        model.model_messages = None;
    }
    Ok(model)
}

/// Builds a provider-neutral model descriptor from an explicit allowlist of
/// local-only metadata.
///
/// A manifest may reuse the slug of a bundled OpenAI model, but that must not
/// opt a custom provider into bundled wire-format capabilities such as
/// Responses Lite. Local prompt text remains a Codex-owned decision, so known
/// bundled prompts are safe to preserve.
fn safe_provider_manifest_model_info(slug: &str, bundled_model: Option<&ModelInfo>) -> ModelInfo {
    ModelInfo {
        slug: slug.to_string(),
        display_name: slug.to_string(),
        description: None,
        default_reasoning_level: None,
        supported_reasoning_levels: Vec::new(),
        shell_type: ConfigShellToolType::Default,
        visibility: ModelVisibility::None,
        supported_in_api: true,
        priority: 99,
        additional_speed_tiers: Vec::new(),
        service_tiers: Vec::new(),
        default_service_tier: None,
        availability_nux: None,
        upgrade: None,
        base_instructions: bundled_model.map_or_else(
            || BASE_INSTRUCTIONS.to_string(),
            |model| model.base_instructions.clone(),
        ),
        model_messages: bundled_model.and_then(|model| model.model_messages.clone()),
        include_skills_usage_instructions: false,
        supports_reasoning_summary_parameter: true,
        default_reasoning_summary: ReasoningSummary::Auto,
        support_verbosity: false,
        default_verbosity: None,
        apply_patch_tool_type: None,
        web_search_tool_type: WebSearchToolType::Text,
        truncation_policy: TruncationPolicyConfig::bytes(/*limit*/ 10_000),
        supports_parallel_tool_calls: false,
        supports_image_detail_original: false,
        context_window: None,
        max_context_window: None,
        auto_compact_token_limit: None,
        comp_hash: None,
        effective_context_window_percent: 95,
        experimental_supported_tools: Vec::new(),
        input_modalities: default_input_modalities(),
        used_fallback_model_metadata: false,
        supports_search_tool: false,
        use_responses_lite: false,
        auto_review_model_override: None,
        tool_mode: None,
        multi_agent_version: None,
    }
}

fn positive_limit(name: &str, value: Option<i64>) -> Result<Option<i64>, String> {
    if value.is_some_and(|value| value <= 0) {
        return Err(format!("provider manifest model {name} must be positive"));
    }
    if value.is_some_and(|value| value > MAX_SAFE_MODEL_TOKEN_LIMIT) {
        return Err(format!(
            "provider manifest model {name} must be no greater than {MAX_SAFE_MODEL_TOKEN_LIMIT}"
        ));
    }
    Ok(value)
}

fn reasoning_preset(model: &ModelInfo, effort: ReasoningEffort) -> ReasoningEffortPreset {
    model
        .supported_reasoning_levels
        .iter()
        .find(|preset| preset.effort == effort)
        .cloned()
        .unwrap_or_else(|| ReasoningEffortPreset {
            description: effort.to_string(),
            effort,
        })
}

fn to_service_tier(service_tier: ProviderManifestServiceTier) -> Result<ModelServiceTier, String> {
    if service_tier.id.is_empty()
        || service_tier.id.trim() != service_tier.id
        || service_tier.name.is_empty()
        || service_tier.name.trim() != service_tier.name
    {
        return Err("provider manifest service tiers require non-empty id and name".to_string());
    }
    Ok(ModelServiceTier {
        id: service_tier.id,
        name: service_tier.name,
        description: service_tier.description,
    })
}

#[cfg(test)]
#[path = "provider_manifest_tests.rs"]
mod tests;
