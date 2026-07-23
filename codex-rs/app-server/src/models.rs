use std::sync::Arc;

use codex_app_server_protocol::Model;
use codex_app_server_protocol::ModelServiceTier;
use codex_app_server_protocol::ModelUpgradeInfo;
use codex_app_server_protocol::ReasoningEffortOption;
use codex_core::ThreadManager;
use codex_http_client::HttpClientFactory;
use codex_models_manager::manager::RefreshStrategy;
use codex_protocol::ThreadId;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::openai_models::ModelPreset;
use codex_protocol::openai_models::ReasoningEffortPreset;

pub async fn supported_models(
    thread_manager: Arc<ThreadManager>,
    thread_id: Option<ThreadId>,
    include_hidden: bool,
    http_client_factory: HttpClientFactory,
) -> CodexResult<Vec<Model>> {
    let presets = match thread_id {
        Some(thread_id) => {
            thread_manager
                .list_models_for_thread(thread_id, RefreshStrategy::OnlineIfUncached)
                .await?
        }
        None => {
            thread_manager
                .list_models_with_refresh_error(
                    RefreshStrategy::OnlineIfUncached,
                    http_client_factory,
                )
                .await?
        }
    };
    Ok(presets
        .into_iter()
        .filter(|preset| include_hidden || preset.show_in_picker)
        .map(model_from_preset)
        .collect())
}

fn model_from_preset(preset: ModelPreset) -> Model {
    Model {
        id: preset.id.to_string(),
        model: preset.model.to_string(),
        upgrade: preset.upgrade.as_ref().map(|upgrade| upgrade.id.clone()),
        upgrade_info: preset.upgrade.as_ref().map(|upgrade| ModelUpgradeInfo {
            model: upgrade.id.clone(),
            upgrade_copy: upgrade.upgrade_copy.clone(),
            model_link: upgrade.model_link.clone(),
            migration_markdown: upgrade.migration_markdown.clone(),
        }),
        availability_nux: preset.availability_nux.map(Into::into),
        display_name: preset.display_name.to_string(),
        description: preset.description.to_string(),
        hidden: !preset.show_in_picker,
        supported_reasoning_efforts: reasoning_efforts_from_preset(
            preset.supported_reasoning_efforts,
        ),
        default_reasoning_effort: preset.default_reasoning_effort,
        input_modalities: preset.input_modalities,
        supports_personality: preset.supports_personality,
        additional_speed_tiers: preset.additional_speed_tiers,
        service_tiers: preset
            .service_tiers
            .into_iter()
            .map(|service_tier| ModelServiceTier {
                id: service_tier.id,
                name: service_tier.name,
                description: service_tier.description,
            })
            .collect(),
        default_service_tier: preset.default_service_tier,
        is_default: preset.is_default,
    }
}

fn reasoning_efforts_from_preset(
    efforts: Vec<ReasoningEffortPreset>,
) -> Vec<ReasoningEffortOption> {
    efforts
        .into_iter()
        .map(|preset| ReasoningEffortOption {
            reasoning_effort: preset.effort,
            description: preset.description,
        })
        .collect()
}
