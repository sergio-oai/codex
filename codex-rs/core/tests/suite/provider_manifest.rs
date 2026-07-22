#![cfg(not(target_os = "windows"))]

use anyhow::Result;
use codex_protocol::config_types::ServiceTier;
use core_test_support::responses::mount_sse_once;
use core_test_support::responses::sse_completed;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::json;
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
fn provider_manifest_outage_without_configured_model_fails_startup() -> Result<()> {
    run_provider_manifest_test(
        "provider_manifest_outage_without_configured_model_fails_startup",
        || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            runtime.block_on(provider_manifest_outage_without_configured_model_fails_startup_impl())
        },
    )
}

async fn provider_manifest_outage_without_configured_model_fails_startup_impl() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/codex/provider-manifest"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;

    let mut builder = test_codex().with_config(|config| {
        config.model = None;
        config.model_provider.provider_manifest_path = Some("codex/provider-manifest".to_string());
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
        error
            .to_string()
            .contains("provider manifest codex/provider-manifest did not yield an available model"),
        "unexpected startup error: {error:#}"
    );

    let requests = server
        .received_requests()
        .await
        .expect("mock server should capture requests");
    assert!(
        requests
            .iter()
            .any(|request| request.url.path() == "/v1/codex/provider-manifest"),
        "startup should try the configured provider manifest before failing"
    );

    server.verify().await;
    Ok(())
}
