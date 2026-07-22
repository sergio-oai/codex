use codex_protocol::openai_models::ModelPreset;
use codex_protocol::openai_models::ModelVisibility;
use codex_protocol::openai_models::ReasoningEffort;
use pretty_assertions::assert_eq;
use serde_json::json;

use super::parse_provider_manifest;

#[test]
fn parses_safe_model_metadata_and_clears_bundled_service_tiers() {
    let body = serde_json::to_vec(&json!({
        "schema_version": 1,
        "provider": {
            "id": "venado",
            "display_name": "Venado"
        },
        "features": {
            "service_tiers": []
        },
        "models": [{
            "id": "gpt-5.4",
            "display_name": "GPT-5.4 on Venado",
            "description": "Provider-hosted GPT-5.4",
            "context_window": 190000,
            "max_input_tokens": 163200,
            "max_output_tokens": 24000,
            "default_reasoning_effort": "medium",
            "supported_reasoning_efforts": ["low", "medium", "high"],
            "supports_personality": false,
            "service_tiers": [],
            "base_instructions": "do not trust remote instructions"
        }]
    }))
    .expect("manifest serializes");

    let models = parse_provider_manifest(&body).expect("manifest parses");
    let model = models.first().expect("one manifest model");

    assert_eq!(models.len(), 1);
    assert_eq!(model.slug, "gpt-5.4");
    assert_eq!(model.display_name, "GPT-5.4 on Venado");
    assert_eq!(
        model.description.as_deref(),
        Some("Provider-hosted GPT-5.4")
    );
    assert_eq!(model.context_window, Some(163_200));
    assert_eq!(model.max_context_window, Some(163_200));
    assert_eq!(model.default_reasoning_level, Some(ReasoningEffort::Medium));
    assert_eq!(model.visibility, ModelVisibility::List);
    assert_eq!(model.service_tiers, Vec::new());
    assert_eq!(model.additional_speed_tiers, Vec::<String>::new());
    assert_eq!(model.default_service_tier, None);
    assert_eq!(model.availability_nux, None);
    assert_eq!(model.upgrade, None);
    assert_eq!(model.auto_review_model_override, None);
    assert_eq!(model.model_messages, None);
    assert_ne!(model.base_instructions, "do not trust remote instructions");
    assert_eq!(
        model.service_tier_for_request(Some("priority".to_string())),
        None
    );
    assert!(!ModelPreset::from(model.clone()).supports_fast_mode());
}

#[test]
fn uses_max_input_tokens_when_context_window_is_unknown() {
    let body = serde_json::to_vec(&json!({
        "schema_version": 1,
        "models": [{
            "id": "d8",
            "display_name": "d8",
            "description": "",
            "context_window": null,
            "max_input_tokens": 16384,
            "default_reasoning_effort": null,
            "supported_reasoning_efforts": [],
            "service_tiers": []
        }]
    }))
    .expect("manifest serializes");

    let models = parse_provider_manifest(&body).expect("manifest parses");

    assert_eq!(models[0].context_window, Some(16_384));
    assert_eq!(models[0].max_context_window, Some(16_384));
}

#[test]
fn rejects_unsupported_schema_versions_and_duplicate_models() {
    let unsupported = serde_json::to_vec(&json!({
        "schema_version": 2,
        "models": []
    }))
    .expect("manifest serializes");
    let duplicate = serde_json::to_vec(&json!({
        "schema_version": 1,
        "models": [
            {
                "id": "d8",
                "display_name": "d8",
                "service_tiers": []
            },
            {
                "id": "d8",
                "display_name": "d8",
                "service_tiers": []
            }
        ]
    }))
    .expect("manifest serializes");
    let unsupported_default = serde_json::to_vec(&json!({
        "schema_version": 1,
        "models": [{
            "id": "d8",
            "display_name": "d8",
            "default_reasoning_effort": "high",
            "supported_reasoning_efforts": ["low"],
            "service_tiers": []
        }]
    }))
    .expect("manifest serializes");

    assert!(
        parse_provider_manifest(&unsupported)
            .expect_err("new schema should fail")
            .contains("unsupported provider manifest schema_version")
    );
    assert!(
        parse_provider_manifest(&duplicate)
            .expect_err("duplicate model should fail")
            .contains("duplicate model id")
    );
    assert!(
        parse_provider_manifest(&unsupported_default)
            .expect_err("unsupported default should fail")
            .contains("unsupported default_reasoning_effort")
    );
}
