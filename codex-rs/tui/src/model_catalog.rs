use codex_protocol::openai_models::ModelPreset;
use std::convert::Infallible;

/// Snapshot of the provider kind that produced a model catalog.
///
/// Keep this separate from mutable runtime config: users can edit a provider
/// definition while the TUI is open, and a remote thread can use a different
/// provider than the process-level startup config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelCatalogProvenance {
    KnownOrdinary,
    KnownManifest,
    Unknown,
}

#[derive(Debug, Clone)]
pub(crate) struct ModelCatalog {
    models: Vec<ModelPreset>,
}

impl ModelCatalog {
    pub(crate) fn new(models: Vec<ModelPreset>) -> Self {
        Self { models }
    }

    pub(crate) fn try_list_models(&self) -> Result<Vec<ModelPreset>, Infallible> {
        Ok(self.models.clone())
    }
}
