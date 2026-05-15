use llmfit_core::hardware::SystemSpecs;
use serde::Serialize;
use std::sync::Arc;

use crate::serve_shared;

pub struct EventPublisher {
    client: async_nats::Client,
    hostname: String,
}

#[derive(Serialize)]
struct EventEnvelope<'a, T: Serialize> {
    timestamp: String,
    hostname: &'a str,
    event_type: &'a str,
    version: &'static str,
    data: &'a T,
}

impl EventPublisher {
    pub async fn connect(nats_url: &str) -> Result<Self, String> {
        let client = async_nats::connect(nats_url)
            .await
            .map_err(|e| format!("NATS connection failed: {e}"))?;

        let hostname = std::env::var("HOSTNAME")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| "unknown-node".to_string());

        Ok(Self { client, hostname })
    }

    pub async fn publish_system(&self, specs: &SystemSpecs) {
        let data = serve_shared::system_json(specs);
        self.publish("system", &data).await;
    }

    pub async fn publish_event<T: Serialize>(&self, event_type: &str, data: &T) {
        self.publish(event_type, data).await;
    }

    async fn publish<T: Serialize>(&self, event_type: &str, data: &T) {
        let subject = format!("llmfit.{}.{}", event_type, self.hostname);
        let envelope = EventEnvelope {
            timestamp: chrono_now(),
            hostname: &self.hostname,
            event_type,
            version: "1",
            data,
        };

        let payload = match serde_json::to_vec(&envelope) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("[llmfit-events] serialization error: {e}");
                return;
            }
        };

        if let Err(e) = self.client.publish(subject, payload.into()).await {
            eprintln!("[llmfit-events] publish error: {e}");
        }
    }
}

/// Spawn a background task that publishes system specs every 60 seconds.
pub fn start_periodic_publisher(publisher: Arc<EventPublisher>, specs: SystemSpecs) {
    tokio::spawn(async move {
        // Publish immediately on startup
        publisher.publish_system(&specs).await;

        let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
        interval.tick().await; // skip the first (immediate) tick
        loop {
            interval.tick().await;
            publisher.publish_system(&specs).await;
        }
    });
}

fn chrono_now() -> String {
    // Simple ISO 8601 timestamp without pulling in chrono crate
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Format as seconds since epoch — simple and sortable
    format!("{secs}")
}
