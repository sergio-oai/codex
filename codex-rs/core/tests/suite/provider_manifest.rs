#![cfg(not(target_os = "windows"))]

use anyhow::Result;
use codex_core::config::AgentRoleConfig;
use codex_features::Feature;
use codex_protocol::config_types::ServiceTier;
use codex_protocol::openai_models::ReasoningEffort;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_function_call_with_namespace;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::mount_sse_once;
use core_test_support::responses::mount_sse_once_match;
use core_test_support::responses::namespace_child_tool;
use core_test_support::responses::sse;
use core_test_support::responses::sse_completed;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::time::Instant;
use tokio::time::sleep;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

#[test]
fn provider_manifest_selects_model_and_omits_unsupported_fast_tier() -> Result<()> {
    run_provider_manifest_test(
        "provider_manifest_selects_model_and_omits_unsupported_fast_tier",
        || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            runtime.block_on(provider_manifest_selects_model_and_omits_unsupported_fast_tier_impl())
        },
    )
}

fn run_provider_manifest_test(
    name: &str,
    test: impl FnOnce() -> Result<()> + Send + 'static,
) -> Result<()> {
    const TEST_STACK_SIZE_BYTES: usize = 8 * 1024 * 1024;
    let handle = std::thread::Builder::new()
        .name(name.to_string())
        .stack_size(TEST_STACK_SIZE_BYTES)
        .spawn(test)?;

    match handle.join() {
        Ok(result) => result,
        Err(_) => Err(anyhow::anyhow!("provider manifest test thread panicked")),
    }
}

async fn provider_manifest_selects_model_and_omits_unsupported_fast_tier_impl() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/codex/provider-manifest"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "schema_version": 1,
            "models": [{
                "id": "venado-only",
                "display_name": "Venado only",
                "description": "Only available from the provider manifest",
                "context_window": 190_000,
                "max_input_tokens": 163_200,
                "default_reasoning_effort": "medium",
                "supported_reasoning_efforts": ["low", "medium", "high"],
                "supports_personality": false,
                "service_tiers": []
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;
    let responses_mock = mount_sse_once(&server, sse_completed("resp-1")).await;

    let mut builder = test_codex().with_config(|config| {
        config.model = None;
        config.model_provider.provider_manifest_path = Some("codex/provider-manifest".to_string());
        config.service_tier = Some(ServiceTier::Fast.request_value().to_string());
    });
    let test = builder.build(&server).await?;

    assert_eq!(test.session_configured.model, "venado-only");

    test.submit_turn("hello from Catalyst").await?;

    let response = responses_mock.single_request();
    let body = response.body_json();
    assert_eq!(body["model"].as_str(), Some("venado-only"));
    assert_eq!(body.get("service_tier"), None);

    let requests = server
        .received_requests()
        .await
        .expect("mock server should capture requests");
    assert_eq!(
        requests
            .iter()
            .filter(|request| request.url.path() == "/v1/codex/provider-manifest")
            .count(),
        1
    );
    assert!(
        requests
            .iter()
            .all(|request| request.url.path() != "/v1/models"),
        "an opted-in provider manifest should replace the ordinary /models request"
    );

    server.verify().await;
    Ok(())
}

#[test]
fn provider_manifest_custom_model_without_marker_is_visible_to_multi_agent_v2() -> Result<()> {
    run_provider_manifest_test(
        "provider_manifest_custom_model_without_marker_is_visible_to_multi_agent_v2",
        || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            runtime.block_on(
                provider_manifest_custom_model_without_marker_is_visible_to_multi_agent_v2_impl(),
            )
        },
    )
}

async fn provider_manifest_custom_model_without_marker_is_visible_to_multi_agent_v2_impl()
-> Result<()> {
    skip_if_no_network!(Ok(()));

    const MANIFEST_MODEL: &str = "venado-only";
    const MULTI_AGENT_V2_NAMESPACE: &str = "collaboration";

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/codex/provider-manifest"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "schema_version": 1,
            "models": [{
                "id": MANIFEST_MODEL,
                "display_name": "Venado only",
                "context_window": 32_000,
                "max_input_tokens": 24_000,
                "service_tiers": []
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;
    let responses_mock = mount_sse_once(&server, sse_completed("resp-v2-1")).await;

    let mut builder = test_codex().with_config(|config| {
        config.model = None;
        config.model_provider.provider_manifest_path = Some("codex/provider-manifest".to_string());
        config
            .features
            .enable(Feature::Collab)
            .expect("test config should allow feature update");
        config
            .features
            .enable(Feature::MultiAgentV2)
            .expect("test config should allow feature update");
        config.multi_agent_v2.expose_spawn_agent_model_overrides = true;
    });
    let test = builder.build(&server).await?;

    test.submit_turn("show me the available model overrides")
        .await?;

    let body = responses_mock.single_request().body_json();
    let description = namespace_child_tool(&body, MULTI_AGENT_V2_NAMESPACE, "spawn_agent")
        .and_then(|tool| tool.get("description"))
        .and_then(serde_json::Value::as_str)
        .expect("v2 spawn_agent description should be present");
    assert!(
        description.contains(MANIFEST_MODEL),
        "custom manifest model should be visible to v2 spawn_agent: {description:?}"
    );

    server.verify().await;
    Ok(())
}

#[test]
fn provider_manifest_rejects_unadvertised_configured_model() -> Result<()> {
    run_provider_manifest_test(
        "provider_manifest_rejects_unadvertised_configured_model",
        || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            runtime.block_on(provider_manifest_rejects_unadvertised_configured_model_impl())
        },
    )
}

async fn provider_manifest_rejects_unadvertised_configured_model_impl() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/codex/provider-manifest"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "schema_version": 1,
            "models": [{
                "id": "venado-only",
                "display_name": "Venado only",
                "max_input_tokens": 24_000,
                "service_tiers": []
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let mut builder = test_codex().with_config(|config| {
        config.model = Some("not-advertised".to_string());
        config.model_provider.provider_manifest_path = Some("codex/provider-manifest".to_string());
    });
    let error = match builder.build(&server).await {
        Ok(_) => {
            return Err(anyhow::anyhow!(
                "session startup unexpectedly accepted an unadvertised manifest model"
            ));
        }
        Err(error) => error,
    };

    assert!(
        error.to_string().contains(
            "provider manifest codex/provider-manifest does not advertise configured model not-advertised"
        ),
        "unexpected startup error: {error:#}"
    );

    server.verify().await;
    Ok(())
}

#[test]
fn provider_manifest_rejects_unsupported_configured_reasoning_effort() -> Result<()> {
    run_provider_manifest_test(
        "provider_manifest_rejects_unsupported_configured_reasoning_effort",
        || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            runtime
                .block_on(provider_manifest_rejects_unsupported_configured_reasoning_effort_impl())
        },
    )
}

async fn provider_manifest_rejects_unsupported_configured_reasoning_effort_impl() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/codex/provider-manifest"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "schema_version": 1,
            "models": [{
                "id": "venado-only",
                "display_name": "Venado only",
                "max_input_tokens": 24_000,
                "default_reasoning_effort": "medium",
                "supported_reasoning_efforts": ["medium"],
                "service_tiers": []
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let mut builder = test_codex().with_config(|config| {
        config.model = Some("venado-only".to_string());
        config.model_reasoning_effort = Some(ReasoningEffort::High);
        config.model_provider.provider_manifest_path = Some("codex/provider-manifest".to_string());
    });
    let error = match builder.build(&server).await {
        Ok(_) => {
            return Err(anyhow::anyhow!(
                "session startup unexpectedly accepted an unsupported manifest reasoning effort"
            ));
        }
        Err(error) => error,
    };

    assert!(
        error.to_string().contains(
            "provider manifest codex/provider-manifest model venado-only does not advertise configured reasoning effort high"
        ),
        "unexpected startup error: {error:#}"
    );

    server.verify().await;
    Ok(())
}

#[test]
fn provider_manifest_outage_fails_startup_with_or_without_configured_model() -> Result<()> {
    run_provider_manifest_test(
        "provider_manifest_outage_fails_startup_with_or_without_configured_model",
        || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            runtime.block_on(
                provider_manifest_outage_fails_startup_with_or_without_configured_model_impl(),
            )
        },
    )
}

async fn provider_manifest_outage_fails_startup_with_or_without_configured_model_impl() -> Result<()>
{
    skip_if_no_network!(Ok(()));

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/codex/provider-manifest"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;

    for configured_model in [None, Some("d8")] {
        let mut builder = test_codex().with_config(move |config| {
            config.model = configured_model.map(str::to_string);
            config.model_provider.provider_manifest_path =
                Some("codex/provider-manifest".to_string());
        });
        let error = match builder.build(&server).await {
            Ok(_) => {
                return Err(anyhow::anyhow!(
                    "session startup unexpectedly succeeded without a manifest catalog"
                ));
            }
            Err(error) => error,
        };

        assert!(
            error.to_string().contains(
                "provider manifest codex/provider-manifest did not yield an available model"
            ),
            "unexpected startup error: {error:#}"
        );
    }

    let requests = server
        .received_requests()
        .await
        .expect("mock server should capture requests");
    assert!(
        requests
            .iter()
            .filter(|request| request.url.path() == "/v1/codex/provider-manifest")
            .count()
            >= 1,
        "startup should try the configured provider manifest before failing"
    );

    server.verify().await;
    Ok(())
}

#[test]
fn provider_manifest_startup_uses_successful_retry_catalog() -> Result<()> {
    run_provider_manifest_test(
        "provider_manifest_startup_uses_successful_retry_catalog",
        || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            runtime.block_on(provider_manifest_startup_uses_successful_retry_catalog_impl())
        },
    )
}

async fn provider_manifest_startup_uses_successful_retry_catalog_impl() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = MockServer::start().await;
    let request_count = Arc::new(AtomicUsize::new(0));
    let response_count = Arc::clone(&request_count);
    Mock::given(method("GET"))
        .and(path("/v1/codex/provider-manifest"))
        .respond_with(move |_request: &wiremock::Request| {
            if response_count.fetch_add(1, Ordering::SeqCst) == 0 {
                ResponseTemplate::new(500)
            } else {
                ResponseTemplate::new(200).set_body_json(json!({
                    "schema_version": 1,
                    "models": [{
                        "id": "venado-only",
                        "display_name": "Venado only",
                        "max_input_tokens": 24_000,
                        "service_tiers": []
                    }]
                }))
            }
        })
        .expect(2)
        .mount(&server)
        .await;

    let mut builder = test_codex().with_config(|config| {
        config.model = None;
        config.model_provider.provider_manifest_path = Some("codex/provider-manifest".to_string());
    });
    let test = builder.build(&server).await?;

    assert_eq!(test.session_configured.model, "venado-only");
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    server.verify().await;
    Ok(())
}

#[test]
fn spawned_child_role_fetches_unloaded_provider_manifest() -> Result<()> {
    run_provider_manifest_test(
        "spawned_child_role_fetches_unloaded_provider_manifest",
        || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .thread_stack_size(8 * 1024 * 1024)
                .enable_all()
                .build()?;
            runtime.block_on(spawned_child_role_fetches_unloaded_provider_manifest_impl(
                ManifestChildSpawnMode::RequestedModel,
            ))
        },
    )
}

#[test]
fn spawned_child_role_reasoning_only_loads_unloaded_provider_manifest() -> Result<()> {
    run_provider_manifest_test(
        "spawned_child_role_reasoning_only_loads_unloaded_provider_manifest",
        || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .thread_stack_size(8 * 1024 * 1024)
                .enable_all()
                .build()?;
            runtime.block_on(spawned_child_role_fetches_unloaded_provider_manifest_impl(
                ManifestChildSpawnMode::RoleLockedReasoningOnly,
            ))
        },
    )
}

#[derive(Clone, Copy)]
enum ManifestChildSpawnMode {
    RequestedModel,
    RoleLockedReasoningOnly,
}

async fn spawned_child_role_fetches_unloaded_provider_manifest_impl(
    mode: ManifestChildSpawnMode,
) -> Result<()> {
    skip_if_no_network!(Ok(()));

    const PARENT_PROMPT: &str = "spawn the manifest-backed worker";
    const CHILD_PROMPT: &str = "use the manifest-backed model";
    const SPAWN_CALL_ID: &str = "spawn-manifest-worker";
    const ROLE_NAME: &str = "manifest_worker";
    const MANIFEST_MODEL: &str = "venado-only";

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/codex/provider-manifest"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "schema_version": 1,
            "models": [{
                "id": MANIFEST_MODEL,
                "display_name": "Venado only",
                "description": "Only available through the child role provider",
                "context_window": 32_000,
                "max_input_tokens": 24_000,
                "default_reasoning_effort": "medium",
                "supported_reasoning_efforts": ["medium"],
                "multi_agent_version": "v1",
                "service_tiers": [{
                    "id": "priority",
                    "name": "Fast",
                    "description": "Fast provider tier"
                }]
            }]
        })))
        .mount(&server)
        .await;

    let spawn_args = match mode {
        ManifestChildSpawnMode::RequestedModel => serde_json::to_string(&json!({
            "message": CHILD_PROMPT,
            "agent_type": ROLE_NAME,
            "model": MANIFEST_MODEL,
            "service_tier": "priority",
        }))?,
        ManifestChildSpawnMode::RoleLockedReasoningOnly => serde_json::to_string(&json!({
            "message": CHILD_PROMPT,
            "agent_type": ROLE_NAME,
            "reasoning_effort": "medium",
            "service_tier": "priority",
        }))?,
    };
    mount_sse_once_match(
        &server,
        |request: &wiremock::Request| request_body_contains(request, PARENT_PROMPT),
        sse(vec![
            ev_response_created("resp-parent-1"),
            ev_function_call_with_namespace(
                SPAWN_CALL_ID,
                "multi_agent_v1",
                "spawn_agent",
                &spawn_args,
            ),
            ev_completed("resp-parent-1"),
        ]),
    )
    .await;
    let child_response = mount_sse_once_match(
        &server,
        |request: &wiremock::Request| {
            request_has_input_type(request, "agent_message")
                && request_body_contains(request, CHILD_PROMPT)
                && !request_has_input_type(request, "function_call_output")
                && request_uses_model(request, MANIFEST_MODEL)
        },
        sse(vec![
            ev_response_created("resp-child-1"),
            ev_assistant_message("msg-child-1", "child done"),
            ev_completed("resp-child-1"),
        ]),
    )
    .await;
    mount_sse_once_match(
        &server,
        |request: &wiremock::Request| request_body_contains(request, SPAWN_CALL_ID),
        sse(vec![
            ev_response_created("resp-parent-2"),
            ev_assistant_message("msg-parent-2", "worker spawned"),
            ev_completed("resp-parent-2"),
        ]),
    )
    .await;

    let provider_base_url = format!("{}/v1", server.uri());
    let mut builder = test_codex().with_config(move |config| {
        config
            .features
            .enable(Feature::Collab)
            .expect("test config should allow feature update");
        let role_path = config.codex_home.join("manifest-worker-role.toml");
        let role_model = matches!(mode, ManifestChildSpawnMode::RoleLockedReasoningOnly)
            .then_some(format!("model = \"{MANIFEST_MODEL}\"\n"))
            .unwrap_or_default();
        std::fs::write(
            &role_path,
            format!(
                r#"
{role_model}
model_provider = "venado"

[model_providers.venado]
name = "Venado"
base_url = "{provider_base_url}"
env_key = "PATH"
wire_api = "responses"
provider_manifest_path = "codex/provider-manifest"
"#
            ),
        )
        .expect("write manifest-backed worker role config");
        config.agent_roles.insert(
            ROLE_NAME.to_string(),
            AgentRoleConfig {
                description: Some("Manifest-backed worker role".to_string()),
                config_file: Some(role_path.to_path_buf()),
                nickname_candidates: None,
            },
        );
    });
    let test = builder.build_with_auto_env(&server).await?;

    let requests_before_spawn = server
        .received_requests()
        .await
        .expect("mock server should capture requests");
    assert!(
        requests_before_spawn
            .iter()
            .all(|request| request.url.path() != "/v1/codex/provider-manifest"),
        "the child-only manifest should remain unloaded during root startup"
    );

    test.submit_turn(PARENT_PROMPT).await?;

    let deadline = Instant::now() + Duration::from_secs(5);
    let child_request = loop {
        if let Some(request) = child_response
            .requests()
            .into_iter()
            .find(|request| request.body_json()["model"].as_str() == Some(MANIFEST_MODEL))
        {
            break request;
        }
        if Instant::now() >= deadline {
            anyhow::bail!("timed out waiting for the manifest-backed child request");
        }
        sleep(Duration::from_millis(10)).await;
    };
    let child_thread_id = test
        .thread_manager
        .list_thread_ids()
        .await
        .into_iter()
        .find(|thread_id| *thread_id != test.session_configured.thread_id)
        .expect("spawned child thread id");
    let child_snapshot = test
        .thread_manager
        .get_thread(child_thread_id)
        .await?
        .config_snapshot()
        .await;
    assert_eq!(child_snapshot.model, MANIFEST_MODEL);
    assert_eq!(child_snapshot.model_provider_id, "venado");
    assert_eq!(child_snapshot.service_tier.as_deref(), Some("priority"));
    let requests_after_spawn = server
        .received_requests()
        .await
        .expect("mock server should capture requests");
    assert!(
        requests_after_spawn
            .iter()
            .any(|request| request.url.path() == "/v1/codex/provider-manifest"),
        "spawning the role should fetch its previously unloaded manifest"
    );
    assert_eq!(
        child_request.body_json()["model"].as_str(),
        Some(MANIFEST_MODEL)
    );
    assert_eq!(
        child_request.body_json()["service_tier"].as_str(),
        Some("priority")
    );
    if matches!(mode, ManifestChildSpawnMode::RoleLockedReasoningOnly) {
        assert_eq!(
            child_request.body_json()["reasoning"]["effort"].as_str(),
            Some("medium")
        );
    }

    server.verify().await;
    Ok(())
}

fn request_body_contains(request: &wiremock::Request, text: &str) -> bool {
    decoded_request_body(request)
        .and_then(|body| String::from_utf8(body).ok())
        .is_some_and(|body| body.contains(text))
}

fn request_has_input_type(request: &wiremock::Request, input_type: &str) -> bool {
    decoded_request_body(request)
        .and_then(|body| serde_json::from_slice::<serde_json::Value>(&body).ok())
        .and_then(|body| {
            body.get("input")
                .and_then(serde_json::Value::as_array)
                .cloned()
        })
        .is_some_and(|items| {
            items.iter().any(|item| {
                item.get("type").and_then(serde_json::Value::as_str) == Some(input_type)
            })
        })
}

fn request_uses_model(request: &wiremock::Request, model: &str) -> bool {
    decoded_request_body(request)
        .and_then(|body| serde_json::from_slice::<serde_json::Value>(&body).ok())
        .and_then(|body| {
            body.get("model")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })
        .as_deref()
        == Some(model)
}

fn decoded_request_body(request: &wiremock::Request) -> Option<Vec<u8>> {
    let is_zstd = request
        .headers
        .get("content-encoding")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value
                .split(',')
                .any(|entry| entry.trim().eq_ignore_ascii_case("zstd"))
        });
    if is_zstd {
        zstd::stream::decode_all(std::io::Cursor::new(&request.body)).ok()
    } else {
        Some(request.body.clone())
    }
}
