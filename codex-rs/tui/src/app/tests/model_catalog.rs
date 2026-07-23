use super::*;
use assert_matches::assert_matches;
use codex_config::types::ModelAvailabilityNuxConfig;
use codex_protocol::openai_models::ModelAvailabilityNux;
use pretty_assertions::assert_eq;
use tokio::sync::mpsc::unbounded_channel;

fn all_model_presets() -> Vec<ModelPreset> {
    crate::test_support::TEST_MODEL_PRESETS.clone()
}

fn model_presets_with_test_upgrades() -> Vec<ModelPreset> {
    let mut presets = all_model_presets();
    for model in ["gpt-5.2", "gpt-5.4"] {
        let preset = presets
            .iter_mut()
            .find(|preset| preset.model == model)
            .unwrap_or_else(|| panic!("{model} preset present"));
        preset.upgrade = Some(ModelUpgrade {
            id: "gpt-5.5".to_string(),
            migration_config_key: "hide_test_migration_prompt".to_string(),
            model_link: None,
            upgrade_copy: None,
            migration_markdown: None,
        });
    }
    presets
}

fn model_availability_nux_config(shown_count: &[(&str, u32)]) -> ModelAvailabilityNuxConfig {
    ModelAvailabilityNuxConfig {
        shown_count: shown_count
            .iter()
            .map(|(model, count)| ((*model).to_string(), *count))
            .collect(),
    }
}

#[tokio::test]
async fn thread_catalog_refresh_preserves_ordinary_local_behavior() {
    let app = make_test_app().await;
    let config = app.config.clone();
    let ordinary_provider_id = config.model_provider_id.clone();

    assert!(
        !should_refresh_thread_model_catalog(
            ModelCatalogProvenance::KnownOrdinary,
            effective_thread_provider_uses_manifest(
                &config,
                ordinary_provider_id.as_str(),
                /*reported_uses_manifest*/ None,
            ),
        ),
        "known ordinary providers keep the startup-only model listing path"
    );
    assert!(
        should_refresh_thread_model_catalog(
            ModelCatalogProvenance::KnownOrdinary,
            effective_thread_provider_uses_manifest(
                &config,
                "restored-provider",
                /*reported_uses_manifest*/ None,
            ),
        ),
        "a provider restored outside the local config must be treated as unknown"
    );
    assert!(
        should_refresh_thread_model_catalog(
            ModelCatalogProvenance::Unknown,
            effective_thread_provider_uses_manifest(
                &config,
                ordinary_provider_id.as_str(),
                /*reported_uses_manifest*/ None,
            ),
        ),
        "unknown catalog provenance must be refreshed before reuse"
    );

    let mut manifest_config = config.clone();
    let mut manifest_provider = manifest_config.model_provider.clone();
    manifest_provider.provider_manifest_path = Some("codex/provider-manifest".to_string());
    manifest_config
        .model_providers
        .insert("manifest".to_string(), manifest_provider);
    assert!(
        should_refresh_thread_model_catalog(
            ModelCatalogProvenance::KnownOrdinary,
            effective_thread_provider_uses_manifest(
                &manifest_config,
                "manifest",
                /*reported_uses_manifest*/ None,
            ),
        ),
        "switching to a manifest-backed target needs its thread catalog"
    );
    assert!(
        should_refresh_thread_model_catalog(
            ModelCatalogProvenance::KnownManifest,
            effective_thread_provider_uses_manifest(
                &config,
                ordinary_provider_id.as_str(),
                /*reported_uses_manifest*/ None,
            ),
        ),
        "a same-ID config reload cannot reclassify a manifest-backed current catalog as ordinary"
    );
    assert!(
        should_refresh_thread_model_catalog(
            ModelCatalogProvenance::KnownOrdinary,
            effective_thread_provider_uses_manifest(
                &config,
                ordinary_provider_id.as_str(),
                /*reported_uses_manifest*/ Some(true),
            ),
        ),
        "the loaded thread's effective manifest hint must win over a same-ID ordinary local provider"
    );
    assert!(
        !should_refresh_thread_model_catalog(
            ModelCatalogProvenance::KnownOrdinary,
            effective_thread_provider_uses_manifest(
                &manifest_config,
                "manifest",
                /*reported_uses_manifest*/ Some(false),
            ),
        ),
        "the loaded thread's ordinary-provider hint must win over a same-ID local manifest"
    );
}

#[tokio::test]
async fn loaded_thread_catalog_refresh_keeps_server_manifest_hint() {
    let mut app = make_test_app().await;
    let thread_id = ThreadId::new();
    let channel = app.ensure_thread_channel(thread_id);
    channel.model_provider_uses_manifest = Some(true);
    let store = Arc::clone(&channel.store);
    {
        let mut store = store.lock().await;
        store.set_session(
            super::test_thread_session(thread_id, test_path_buf("/tmp/thread")),
            Vec::new(),
        );
    }

    assert_eq!(
        app.loaded_thread_model_provider(thread_id).await,
        Some(("test-provider".to_string(), Some(true)))
    );
}

#[tokio::test]
async fn bootstrap_catalog_provenance_prefers_server_hint() {
    let app = make_test_app().await;
    let config = app.config.clone();
    let mut manifest_config = config.clone();
    manifest_config.model_provider.provider_manifest_path =
        Some("codex/provider-manifest".to_string());

    assert_eq!(
        bootstrap_model_catalog_provenance(
            &config,
            ThreadParamsMode::Remote,
            /*reported_uses_manifest*/ Some(true),
        ),
        ModelCatalogProvenance::KnownManifest
    );
    assert_eq!(
        bootstrap_model_catalog_provenance(
            &manifest_config,
            ThreadParamsMode::Remote,
            /*reported_uses_manifest*/ Some(false),
        ),
        ModelCatalogProvenance::KnownOrdinary
    );
    assert_eq!(
        bootstrap_model_catalog_provenance(
            &config,
            ThreadParamsMode::Remote,
            /*reported_uses_manifest*/ None,
        ),
        ModelCatalogProvenance::KnownOrdinary
    );
    assert_eq!(
        bootstrap_model_catalog_provenance(
            &manifest_config,
            ThreadParamsMode::Remote,
            /*reported_uses_manifest*/ None,
        ),
        ModelCatalogProvenance::Unknown
    );
    assert_eq!(
        bootstrap_model_catalog_provenance(
            &manifest_config,
            ThreadParamsMode::Embedded,
            /*reported_uses_manifest*/ None,
        ),
        ModelCatalogProvenance::KnownManifest
    );
}

#[tokio::test]
async fn chatwidget_init_carries_manifest_catalog_provenance() {
    let mut app = make_test_app().await;
    app.model_catalog_provenance = ModelCatalogProvenance::KnownManifest;
    let mut tui = crate::tui::test_support::make_test_tui().expect("test tui");

    let init = app.chatwidget_init_for_forked_or_resumed_thread(
        &mut tui,
        app.config.clone(),
        /*initial_user_message*/ None,
    );

    assert_eq!(
        init.model_catalog_provenance,
        ModelCatalogProvenance::KnownManifest
    );
}

fn model_migration_copy_to_plain_text(copy: &crate::model_migration::ModelMigrationCopy) -> String {
    if let Some(markdown) = copy.markdown.as_ref() {
        return markdown.clone();
    }
    let mut s = String::new();
    for span in &copy.heading {
        s.push_str(&span.content);
    }
    s.push('\n');
    s.push('\n');
    for line in &copy.content {
        for span in &line.spans {
            s.push_str(&span.content);
        }
        s.push('\n');
    }
    s
}

#[tokio::test]
async fn model_migration_prompt_only_shows_for_deprecated_models() {
    let seen = BTreeMap::new();
    let presets = model_presets_with_test_upgrades();
    assert!(should_show_model_migration_prompt(
        "gpt-5.2", "gpt-5.5", &seen, &presets
    ));
    assert!(should_show_model_migration_prompt(
        "gpt-5.4", "gpt-5.5", &seen, &presets
    ));
    assert!(!should_show_model_migration_prompt(
        "gpt-5.4", "gpt-5.4", &seen, &presets
    ));
}

#[test]
fn select_model_availability_nux_picks_only_eligible_model() {
    let mut presets = all_model_presets();
    presets.iter_mut().for_each(|preset| {
        preset.availability_nux = None;
    });
    let target = presets
        .iter_mut()
        .find(|preset| preset.model == "gpt-5.4")
        .expect("target preset present");
    target.availability_nux = Some(ModelAvailabilityNux {
        message: "gpt-5.4 is available".to_string(),
    });

    let selected = select_model_availability_nux(&presets, &model_availability_nux_config(&[]));

    assert_eq!(
        selected,
        Some(StartupTooltipOverride {
            model_slug: "gpt-5.4".to_string(),
            message: "gpt-5.4 is available".to_string(),
        })
    );
}

#[test]
fn select_model_availability_nux_does_not_fall_back_to_older_announcement() {
    let mut presets = all_model_presets();
    presets.iter_mut().for_each(|preset| {
        preset.availability_nux = None;
    });
    let gpt_5 = presets
        .iter_mut()
        .find(|preset| preset.model == "gpt-5.4")
        .expect("gpt-5.4 preset present");
    gpt_5.availability_nux = Some(ModelAvailabilityNux {
        message: "gpt-5.4 is available".to_string(),
    });
    let gpt_5_2 = presets
        .iter_mut()
        .find(|preset| preset.model == "gpt-5.4-mini")
        .expect("gpt-5.4-mini preset present");
    gpt_5_2.availability_nux = Some(ModelAvailabilityNux {
        message: "gpt-5.4-mini is available".to_string(),
    });

    let selected = select_model_availability_nux(
        &presets,
        &model_availability_nux_config(&[("gpt-5.4", MODEL_AVAILABILITY_NUX_MAX_SHOW_COUNT)]),
    );

    assert_eq!(selected, None);
}

#[test]
fn select_model_availability_nux_uses_existing_model_order_as_priority() {
    let mut presets = all_model_presets();
    presets.iter_mut().for_each(|preset| {
        preset.availability_nux = None;
    });
    let first = presets
        .iter_mut()
        .find(|preset| preset.model == "gpt-5.4-mini")
        .expect("gpt-5.4-mini preset present");
    first.availability_nux = Some(ModelAvailabilityNux {
        message: "first".to_string(),
    });
    let second = presets
        .iter_mut()
        .find(|preset| preset.model == "gpt-5.4")
        .expect("gpt-5.4 preset present");
    second.availability_nux = Some(ModelAvailabilityNux {
        message: "second".to_string(),
    });

    let selected = select_model_availability_nux(&presets, &model_availability_nux_config(&[]));

    assert_eq!(
        selected,
        Some(StartupTooltipOverride {
            model_slug: "gpt-5.4".to_string(),
            message: "second".to_string(),
        })
    );
}

#[test]
fn select_model_availability_nux_returns_none_when_all_models_are_exhausted() {
    let mut presets = all_model_presets();
    presets.iter_mut().for_each(|preset| {
        preset.availability_nux = None;
    });
    let target = presets
        .iter_mut()
        .find(|preset| preset.model == "gpt-5.4")
        .expect("target preset present");
    target.availability_nux = Some(ModelAvailabilityNux {
        message: "gpt-5.4 is available".to_string(),
    });

    let selected = select_model_availability_nux(
        &presets,
        &model_availability_nux_config(&[("gpt-5.4", MODEL_AVAILABILITY_NUX_MAX_SHOW_COUNT)]),
    );

    assert_eq!(selected, None);
}

#[tokio::test]
async fn prepare_startup_tooltip_override_persists_model_availability_nux_count() {
    let codex_home = tempdir().expect("temp codex home");
    let mut config = ConfigBuilder::default()
        .codex_home(codex_home.path().to_path_buf())
        .build()
        .await
        .expect("config");
    let mut presets = all_model_presets();
    presets.iter_mut().for_each(|preset| {
        preset.availability_nux = None;
    });
    let target = presets
        .iter_mut()
        .find(|preset| preset.model == "gpt-5.4")
        .expect("target preset present");
    target.availability_nux = Some(ModelAvailabilityNux {
        message: "gpt-5.4 is available".to_string(),
    });

    let tooltip =
        prepare_startup_tooltip_override(&mut config, &presets, /*is_first_run*/ false).await;

    assert_eq!(tooltip.as_deref(), Some("gpt-5.4 is available"));
    assert_eq!(
        config.model_availability_nux.shown_count,
        HashMap::from([("gpt-5.4".to_string(), 1)])
    );

    let reloaded = ConfigBuilder::default()
        .codex_home(codex_home.path().to_path_buf())
        .build()
        .await
        .expect("reloaded config");
    assert_eq!(
        reloaded.model_availability_nux.shown_count,
        HashMap::from([("gpt-5.4".to_string(), 1)])
    );
}

#[tokio::test]
async fn accepted_model_migration_persists_target_default_reasoning_effort() {
    let codex_home = tempdir().expect("temp codex home");
    let mut config = ConfigBuilder::default()
        .codex_home(codex_home.path().to_path_buf())
        .build()
        .await
        .expect("config");
    config.model = Some("gpt-5.2".to_string());
    config.model_reasoning_effort = Some(ReasoningEffortConfig::XHigh);

    let (tx_raw, mut rx) = unbounded_channel();
    let app_event_tx = AppEventSender::new(tx_raw);

    apply_accepted_model_migration(
        &mut config,
        &app_event_tx,
        "gpt-5.2".to_string(),
        "gpt-5.4".to_string(),
        ReasoningEffortConfig::Medium,
    );

    assert_eq!(config.model.as_deref(), Some("gpt-5.4"));
    assert_eq!(
        config.model_reasoning_effort,
        Some(ReasoningEffortConfig::Medium)
    );

    let acknowledged = rx.try_recv().expect("acknowledged event");
    assert_matches!(
        acknowledged,
        AppEvent::PersistModelMigrationPromptAcknowledged { from_model, to_model }
            if from_model == "gpt-5.2" && to_model == "gpt-5.4"
    );

    let update_model = rx.try_recv().expect("update model event");
    assert_matches!(
        update_model,
        AppEvent::UpdateModel(model) if model == "gpt-5.4"
    );

    let update_effort = rx.try_recv().expect("update effort event");
    assert_matches!(
        update_effort,
        AppEvent::UpdateReasoningEffort(Some(ReasoningEffortConfig::Medium))
    );

    let persist_selection = rx.try_recv().expect("persist model selection event");
    assert_matches!(
        persist_selection,
        AppEvent::PersistModelSelection { model, effort }
            if model == "gpt-5.4" && effort == Some(ReasoningEffortConfig::Medium)
    );
}

#[tokio::test]
async fn model_migration_prompt_respects_hide_flag_and_self_target() {
    let mut seen = BTreeMap::new();
    seen.insert("gpt-5.2".to_string(), "gpt-5.4".to_string());
    assert!(!should_show_model_migration_prompt(
        "gpt-5.2",
        "gpt-5.4",
        &seen,
        &all_model_presets()
    ));
    assert!(!should_show_model_migration_prompt(
        "gpt-5.4",
        "gpt-5.4",
        &seen,
        &all_model_presets()
    ));
}

#[tokio::test]
async fn model_migration_prompt_skips_when_target_missing_or_hidden() {
    let mut available = all_model_presets();
    let mut current = available
        .iter()
        .find(|preset| preset.model == "gpt-5.2")
        .cloned()
        .expect("preset present");
    current.upgrade = Some(ModelUpgrade {
        id: "missing-target".to_string(),
        migration_config_key: HIDE_GPT5_1_MIGRATION_PROMPT_CONFIG.to_string(),
        model_link: None,
        upgrade_copy: None,
        migration_markdown: None,
    });
    available.retain(|preset| preset.model != "gpt-5.2");
    available.push(current.clone());

    assert!(!should_show_model_migration_prompt(
        &current.model,
        "missing-target",
        &BTreeMap::new(),
        &available,
    ));

    assert!(target_preset_for_upgrade(&available, "missing-target").is_none());

    let mut with_hidden_target = all_model_presets();
    let target = with_hidden_target
        .iter_mut()
        .find(|preset| preset.model == "gpt-5.4")
        .expect("target preset present");
    target.show_in_picker = false;

    assert!(!should_show_model_migration_prompt(
        "gpt-5.2",
        "gpt-5.4",
        &BTreeMap::new(),
        &with_hidden_target,
    ));
    assert!(target_preset_for_upgrade(&with_hidden_target, "gpt-5.4").is_none());
}

#[tokio::test]
async fn model_migration_prompt_shows_for_hidden_model() {
    let codex_home = tempdir().expect("temp codex home");
    let config = ConfigBuilder::default()
        .codex_home(codex_home.path().to_path_buf())
        .build()
        .await
        .expect("config");

    let mut available_models = model_presets_with_test_upgrades();
    let current = available_models
        .iter_mut()
        .find(|preset| preset.model == "gpt-5.2")
        .expect("gpt-5.2 preset present");
    current.show_in_picker = false;
    let current = current.clone();
    assert!(
        !current.show_in_picker,
        "expected gpt-5.2 to be hidden from picker for this test"
    );

    let upgrade = current.upgrade.as_ref().expect("upgrade configured");
    available_models
        .iter_mut()
        .find(|preset| preset.model == upgrade.id)
        .expect("upgrade target present")
        .show_in_picker = true;
    assert!(
        should_show_model_migration_prompt(
            &current.model,
            &upgrade.id,
            &config.notices.model_migrations,
            &available_models,
        ),
        "expected migration prompt to be eligible for hidden model"
    );

    let target =
        target_preset_for_upgrade(&available_models, &upgrade.id).expect("upgrade target present");
    let target_description = (!target.description.is_empty()).then(|| target.description.clone());
    let can_opt_out = true;
    let copy = migration_copy_for_models(
        &current.model,
        &upgrade.id,
        upgrade.model_link.clone(),
        upgrade.upgrade_copy.clone(),
        upgrade.migration_markdown.clone(),
        target.display_name.clone(),
        target_description,
        can_opt_out,
    );

    assert_snapshot!(
        "model_migration_prompt_shows_for_hidden_model",
        model_migration_copy_to_plain_text(&copy)
    );
}
