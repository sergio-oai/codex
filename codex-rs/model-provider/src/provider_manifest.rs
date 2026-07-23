use std::collections::HashSet;

use codex_models_manager::bundled_models_response;
use codex_models_manager::model_info::BASE_INSTRUCTIONS;
use codex_protocol::config_types::ReasoningSummary;
use codex_protocol::config_types::SERVICE_TIER_DEFAULT_REQUEST_VALUE;
use codex_protocol::config_types::ServiceTier;
use codex_protocol::openai_models::ConfigShellToolType;
use codex_protocol::openai_models::InputModality;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ModelServiceTier;
use codex_protocol::openai_models::ModelVisibility;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::openai_models::ReasoningEffortPreset;
use codex_protocol::openai_models::TruncationPolicyConfig;
use codex_protocol::openai_models::WebSearchToolType;
use codex_protocol::protocol::MultiAgentVersion;
use serde::Deserialize;

const SUPPORTED_SCHEMA_VERSION: u32 = 1;
// Provider manifests are fetched from an opt-in remote endpoint. Keep both
// parsing work and any catalog data that can later reach model-visible tool
// descriptions bounded independently of the HTTP client's response limits.
pub(crate) const MAX_PROVIDER_MANIFEST_BYTES: usize = 256 * 1024;
const MAX_PROVIDER_MANIFEST_MODELS: usize = 128;
// spawn_agent exposes at most five picker-visible models. Bound the exact
// manifest-controlled identifiers that it interpolates so even the five
// largest eligible summaries add at most a small, fixed amount of prompt
// context. The remaining punctuation and labels are local Codex text.
const MAX_MODEL_VISIBLE_MANIFEST_MODELS: usize = 5;
const MAX_MODEL_VISIBLE_MANIFEST_BYTES: usize = 512;
const MAX_MODEL_ID_BYTES: usize = 64;
const MAX_MODEL_DISPLAY_NAME_BYTES: usize = 128;
const MAX_MODEL_DESCRIPTION_BYTES: usize = 512;
const MAX_REASONING_EFFORTS_PER_MODEL: usize = 8;
const MAX_REASONING_EFFORT_ID_BYTES: usize = 32;
const MAX_SERVICE_TIERS_PER_MODEL: usize = 4;
const MAX_SERVICE_TIER_ID_BYTES: usize = 32;
const MAX_SERVICE_TIER_NAME_BYTES: usize = 128;
const MAX_SERVICE_TIER_DESCRIPTION_BYTES: usize = 256;
const MAX_INPUT_MODALITIES_PER_MODEL: usize = 3;
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
    /// Modalities accepted by this provider model. Omitted v1 manifests are
    /// text-only by default; image/audio support must be advertised explicitly.
    #[serde(default = "default_provider_manifest_input_modalities")]
    input_modalities: Vec<InputModality>,
    #[serde(default)]
    service_tiers: Vec<ProviderManifestServiceTier>,
    /// Local multi-agent compatibility, not a provider wire-format feature.
    ///
    /// Omitted values inherit this safe local marker from a bundled model with
    /// the same slug. Provider-owned IDs default to v2 so an authoritative
    /// custom catalog remains usable by the current spawn-agent backend without
    /// inheriting OpenAI-only request-shape metadata.
    #[serde(default)]
    multi_agent_version: Option<MultiAgentVersion>,
}

#[derive(Debug, Deserialize)]
struct ProviderManifestServiceTier {
    id: String,
    name: String,
    description: String,
}

pub(crate) fn parse_provider_manifest(body: &[u8]) -> Result<Vec<ModelInfo>, String> {
    if body.len() > MAX_PROVIDER_MANIFEST_BYTES {
        return Err(format!(
            "provider manifest exceeds maximum size of {MAX_PROVIDER_MANIFEST_BYTES} bytes"
        ));
    }
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
    if manifest.models.len() > MAX_PROVIDER_MANIFEST_MODELS {
        return Err(format!(
            "provider manifest must contain no more than {MAX_PROVIDER_MANIFEST_MODELS} models"
        ));
    }

    let bundled_models = bundled_models_response()
        .map_err(|err| format!("failed to load bundled model metadata: {err}"))?
        .models;
    let mut seen_ids = HashSet::new();
    let models = manifest
        .models
        .into_iter()
        .enumerate()
        .map(|(priority, manifest_model)| {
            to_model_info(manifest_model, priority, &bundled_models, &mut seen_ids)
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_model_visible_catalog_budget(&models)?;
    Ok(models)
}

fn to_model_info(
    manifest_model: ProviderManifestModel,
    priority: usize,
    bundled_models: &[ModelInfo],
    seen_ids: &mut HashSet<String>,
) -> Result<ModelInfo, String> {
    validate_model_visible_token("model id", &manifest_model.id, MAX_MODEL_ID_BYTES)?;
    validate_bounded_single_line_text(
        "model display_name",
        &manifest_model.display_name,
        MAX_MODEL_DISPLAY_NAME_BYTES,
    )?;
    validate_bounded_single_line_text(
        "model description",
        &manifest_model.description,
        MAX_MODEL_DESCRIPTION_BYTES,
    )?;
    if !seen_ids.insert(manifest_model.id.clone()) {
        return Err(format!(
            "provider manifest contains duplicate model id {}",
            manifest_model.id
        ));
    }
    validate_reasoning_efforts(&manifest_model)?;
    validate_service_tiers(&manifest_model)?;
    validate_input_modalities(&manifest_model)?;

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
    // Codex uses this value to decide when to compact. Every authoritative
    // manifest model must advertise at least one usable input limit; otherwise
    // the client cannot size or compact requests safely. Use the most
    // conservative known limit so the client compacts before it can send a
    // prompt that the provider rejects.
    let resolved_context_window = match (context_window, max_input_tokens) {
        (Some(context_window), Some(max_input_tokens)) => {
            Some(context_window.min(max_input_tokens))
        }
        (Some(context_window), None) => Some(context_window),
        (None, Some(max_input_tokens)) => Some(max_input_tokens),
        (None, None) => {
            return Err(format!(
                "provider manifest model {} must specify context_window or max_input_tokens",
                manifest_model.id
            ));
        }
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
    // Keep bounded provider prose for user-facing model lists and pickers.
    // The spawn-agent tool renderer suppresses descriptions for
    // manifest-backed turns before building model-visible instructions.
    model.description = Some(manifest_model.description);
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
    model.input_modalities = manifest_model.input_modalities;
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
    // Multi-agent version is a local tool-compatibility marker, unlike
    // Responses Lite or other provider-specific wire capabilities. Explicit
    // manifest values win; exact bundled slugs retain their trusted marker;
    // provider-owned model IDs default to the current v2 backend.
    model.multi_agent_version = manifest_model.multi_agent_version.or_else(|| {
        bundled_model.map_or(Some(MultiAgentVersion::V2), |bundled_model| {
            bundled_model.multi_agent_version
        })
    });
    Ok(model)
}

/// Builds a provider-neutral model descriptor from an explicit allowlist of
/// trusted local-only metadata.
///
/// A manifest may reuse the slug of a bundled OpenAI model, but that must not
/// opt a custom provider into bundled wire-format capabilities such as
/// Responses Lite. For exact bundled-slug matches, preserve Codex-owned prompt,
/// local tool-presentation, and output-truncation behavior that the model was
/// trained to use; request headers, hosted tools, and backend payload
/// capabilities stay conservative.
fn safe_provider_manifest_model_info(slug: &str, bundled_model: Option<&ModelInfo>) -> ModelInfo {
    ModelInfo {
        slug: slug.to_string(),
        display_name: slug.to_string(),
        description: None,
        default_reasoning_level: None,
        supported_reasoning_levels: Vec::new(),
        shell_type: bundled_model.map_or(ConfigShellToolType::Default, |model| model.shell_type),
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
        include_skills_usage_instructions: bundled_model
            .is_some_and(|model| model.include_skills_usage_instructions),
        supports_reasoning_summary_parameter: true,
        default_reasoning_summary: ReasoningSummary::Auto,
        support_verbosity: false,
        default_verbosity: None,
        apply_patch_tool_type: bundled_model.and_then(|model| model.apply_patch_tool_type.clone()),
        web_search_tool_type: WebSearchToolType::Text,
        truncation_policy: bundled_model
            .map_or(TruncationPolicyConfig::bytes(/*limit*/ 10_000), |model| {
                model.truncation_policy
            }),
        supports_parallel_tool_calls: false,
        supports_image_detail_original: false,
        context_window: None,
        max_context_window: None,
        auto_compact_token_limit: None,
        comp_hash: None,
        effective_context_window_percent: bundled_model
            .map_or(95, |model| model.effective_context_window_percent),
        experimental_supported_tools: Vec::new(),
        input_modalities: vec![InputModality::Text],
        used_fallback_model_metadata: false,
        supports_search_tool: false,
        use_responses_lite: false,
        auto_review_model_override: None,
        tool_mode: bundled_model.and_then(|model| model.tool_mode),
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

fn validate_reasoning_efforts(manifest_model: &ProviderManifestModel) -> Result<(), String> {
    if manifest_model.supported_reasoning_efforts.len() > MAX_REASONING_EFFORTS_PER_MODEL {
        return Err(format!(
            "provider manifest model {} must contain no more than {MAX_REASONING_EFFORTS_PER_MODEL} supported reasoning efforts",
            manifest_model.id
        ));
    }

    let mut seen_efforts = HashSet::new();
    for effort in manifest_model
        .default_reasoning_effort
        .iter()
        .chain(manifest_model.supported_reasoning_efforts.iter())
    {
        validate_model_visible_token(
            "reasoning effort id",
            effort.as_str(),
            MAX_REASONING_EFFORT_ID_BYTES,
        )?;
    }
    for effort in &manifest_model.supported_reasoning_efforts {
        if !seen_efforts.insert(effort.as_str()) {
            return Err(format!(
                "provider manifest model {} contains duplicate supported reasoning effort {}",
                manifest_model.id, effort
            ));
        }
    }
    Ok(())
}

fn validate_service_tiers(manifest_model: &ProviderManifestModel) -> Result<(), String> {
    if manifest_model.service_tiers.len() > MAX_SERVICE_TIERS_PER_MODEL {
        return Err(format!(
            "provider manifest model {} must contain no more than {MAX_SERVICE_TIERS_PER_MODEL} service tiers",
            manifest_model.id
        ));
    }

    let mut seen_ids = HashSet::new();
    let mut seen_command_names = HashSet::new();
    for service_tier in &manifest_model.service_tiers {
        validate_model_visible_token(
            "service tier id",
            &service_tier.id,
            MAX_SERVICE_TIER_ID_BYTES,
        )?;
        if service_tier.id == SERVICE_TIER_DEFAULT_REQUEST_VALUE {
            return Err(format!(
                "provider manifest service tier id `{SERVICE_TIER_DEFAULT_REQUEST_VALUE}` is reserved for standard routing"
            ));
        }
        if let Some(canonical_service_tier) = ServiceTier::from_request_value(&service_tier.id)
            && service_tier.id != canonical_service_tier.request_value()
        {
            // Session updates still accept legacy config spellings such as
            // "fast" and canonicalize them to provider request values such as
            // "priority". Manifest IDs are already provider request-contract
            // values, so accepting an alias here would advertise one ID and
            // send another.
            let legacy_service_tier_id = &service_tier.id;
            let canonical_service_tier_id = canonical_service_tier.request_value();
            return Err(format!(
                "provider manifest service tier id `{legacy_service_tier_id}` is reserved for a legacy alias; advertise `{canonical_service_tier_id}` instead"
            ));
        }
        validate_bounded_single_line_text(
            "service tier name",
            &service_tier.name,
            MAX_SERVICE_TIER_NAME_BYTES,
        )?;
        if service_tier.name.is_empty() {
            return Err("provider manifest service tier name must be non-empty".to_string());
        }
        validate_bounded_single_line_text(
            "service tier description",
            &service_tier.description,
            MAX_SERVICE_TIER_DESCRIPTION_BYTES,
        )?;
        if !seen_ids.insert(service_tier.id.as_str()) {
            return Err(format!(
                "provider manifest model {} contains duplicate service tier id {}",
                manifest_model.id, service_tier.id
            ));
        }
        let command_name = service_tier_command_name(&service_tier.id).to_ascii_lowercase();
        if !seen_command_names.insert(command_name.clone()) {
            return Err(format!(
                "provider manifest model {} contains duplicate normalized service tier command {command_name}",
                manifest_model.id
            ));
        }
    }
    Ok(())
}

/// Keeps the remote-controlled part of the spawn_agent model summary small.
///
/// The renderer can select at most five models after applying local
/// multi-agent compatibility filters. Measure the five largest candidates
/// rather than the first five so changing those filters cannot reveal a
/// larger unchecked combination later.
fn validate_model_visible_catalog_budget(models: &[ModelInfo]) -> Result<(), String> {
    let mut model_visible_bytes = models
        .iter()
        .map(model_visible_manifest_bytes)
        .collect::<Vec<_>>();
    model_visible_bytes.sort_unstable_by(|left, right| right.cmp(left));

    let total = model_visible_bytes
        .into_iter()
        .take(MAX_MODEL_VISIBLE_MANIFEST_MODELS)
        .try_fold(0usize, |total, model_bytes| {
            total.checked_add(model_bytes).ok_or_else(|| {
                "provider manifest model-visible metadata size overflowed".to_string()
            })
        })?;
    if total > MAX_MODEL_VISIBLE_MANIFEST_BYTES {
        return Err(format!(
            "provider manifest model-visible metadata must be no more than {MAX_MODEL_VISIBLE_MANIFEST_BYTES} bytes across any {MAX_MODEL_VISIBLE_MANIFEST_MODELS} models"
        ));
    }
    Ok(())
}

fn model_visible_manifest_bytes(model: &ModelInfo) -> usize {
    model.slug.len()
        + model
            .supported_reasoning_levels
            .iter()
            .map(|preset| preset.effort.as_str().len())
            .sum::<usize>()
        + model
            .service_tiers
            .iter()
            .map(|tier| tier.id.len())
            .sum::<usize>()
}

fn default_provider_manifest_input_modalities() -> Vec<InputModality> {
    vec![InputModality::Text]
}

fn validate_input_modalities(manifest_model: &ProviderManifestModel) -> Result<(), String> {
    if manifest_model.input_modalities.is_empty() {
        return Err(format!(
            "provider manifest model {} must contain at least one input modality",
            manifest_model.id
        ));
    }
    if manifest_model.input_modalities.len() > MAX_INPUT_MODALITIES_PER_MODEL {
        return Err(format!(
            "provider manifest model {} must contain no more than {MAX_INPUT_MODALITIES_PER_MODEL} input modalities",
            manifest_model.id
        ));
    }
    let mut seen_modalities = HashSet::new();
    for modality in &manifest_model.input_modalities {
        if !seen_modalities.insert(modality) {
            return Err(format!(
                "provider manifest model {} contains duplicate input modality",
                manifest_model.id
            ));
        }
    }
    if !seen_modalities.contains(&InputModality::Text) {
        return Err(format!(
            "provider manifest model {} must support text input",
            manifest_model.id
        ));
    }
    Ok(())
}

/// Identifiers are interpolated into model-visible model and service-tier
/// summaries. Restrict them to short ASCII tokens so a manifest cannot add
/// quoting, Markdown, newlines, or natural-language instructions there.
fn validate_model_visible_token(name: &str, value: &str, max_bytes: usize) -> Result<(), String> {
    if value.is_empty() || value.trim() != value {
        return Err(format!(
            "provider manifest {name} must be non-empty and trimmed"
        ));
    }
    if value.len() > max_bytes {
        return Err(format!(
            "provider manifest {name} must be no more than {max_bytes} bytes"
        ));
    }
    if !value.bytes().all(|byte| {
        byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/')
    }) {
        return Err(format!(
            "provider manifest {name} must contain only ASCII letters, digits, '-', '_', '.', ':', or '/'"
        ));
    }
    Ok(())
}

/// Bounds UI-only prose and rejects line/control separators. Manifest-backed
/// spawn-agent rendering intentionally omits this prose from model-visible
/// instructions.
fn validate_bounded_single_line_text(
    name: &str,
    value: &str,
    max_bytes: usize,
) -> Result<(), String> {
    if value.len() > max_bytes {
        return Err(format!(
            "provider manifest {name} must be no more than {max_bytes} bytes"
        ));
    }
    if value.trim() != value {
        return Err(format!("provider manifest {name} must be trimmed"));
    }
    if value
        .chars()
        .any(|character| character.is_control() || matches!(character, '\u{2028}' | '\u{2029}'))
    {
        return Err(format!(
            "provider manifest {name} must be single-line text without control characters"
        ));
    }
    Ok(())
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
    // Service-tier names become slash-command identifiers in the TUI. Keep
    // that command identity local: remote display prose is validated above but
    // never allowed to create or shadow a built-in command. Canonical fast and
    // flex IDs retain their existing UX; provider-specific tiers live under a
    // collision-resistant tier- namespace.
    let name = service_tier_command_name(&service_tier.id);
    Ok(ModelServiceTier {
        id: service_tier.id,
        name,
        description: service_tier.description,
    })
}

fn service_tier_command_name(id: &str) -> String {
    match id {
        id if id == ServiceTier::Fast.request_value() => "fast".to_string(),
        id if id == ServiceTier::Flex.request_value() => "flex".to_string(),
        id => format!("tier-{id}"),
    }
}

#[cfg(test)]
#[path = "provider_manifest_tests.rs"]
mod tests;
