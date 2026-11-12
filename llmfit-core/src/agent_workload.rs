//! Context an agent loop grows into.
//!
//! A chat fit assumes the context you declare is roughly the context you use.
//! An agent loop appends every tool call and its result to the transcript, so
//! the KV cache keeps growing while the weights stay put (#914). The profile
//! turns two assumptions the user states, the number of tool turns and the
//! tokens each turn adds, into the context to size memory for.

use crate::fit::DEFAULT_ESTIMATION_CTX;

/// The two assumptions of an agent loop. Neither has a default: they depend on
/// the agent and the task, so the caller states both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentWorkload {
    /// Tool turns the transcript is expected to absorb.
    pub turns: u32,
    /// Tokens one turn appends: the call plus its result.
    pub turn_tokens: u32,
}

impl AgentWorkload {
    /// Tokens of context the loop ends with: the base estimation context plus
    /// what every turn appended. Saturates at `u32::MAX`.
    pub fn context_tokens(&self) -> u32 {
        // u64 holds the product of two u32 values, so only the final
        // narrowing can saturate.
        let grown = u64::from(self.turns) * u64::from(self.turn_tokens);
        u32::try_from(u64::from(DEFAULT_ESTIMATION_CTX) + grown).unwrap_or(u32::MAX)
    }

    /// One line naming the assumptions, so a result never hides them.
    pub fn describe(&self) -> String {
        format!(
            "Agent workload: {} tool turns x {} tokens on a {}-token base = {}-token context for memory estimation",
            self.turns,
            self.turn_tokens,
            DEFAULT_ESTIMATION_CTX,
            self.context_tokens()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_is_the_base_plus_what_every_turn_adds() {
        let w = AgentWorkload {
            turns: 10,
            turn_tokens: 2_048,
        };
        assert_eq!(w.context_tokens(), DEFAULT_ESTIMATION_CTX + 10 * 2_048);
    }

    #[test]
    fn a_single_turn_grows_the_base_by_one_turn() {
        let w = AgentWorkload {
            turns: 1,
            turn_tokens: 1,
        };
        assert_eq!(w.context_tokens(), DEFAULT_ESTIMATION_CTX + 1);
    }

    #[test]
    fn no_turns_leaves_the_base_context() {
        let w = AgentWorkload {
            turns: 0,
            turn_tokens: 4_096,
        };
        assert_eq!(w.context_tokens(), DEFAULT_ESTIMATION_CTX);
    }

    #[test]
    fn the_product_of_two_large_values_saturates_instead_of_wrapping() {
        let w = AgentWorkload {
            turns: u32::MAX,
            turn_tokens: u32::MAX,
        };
        assert_eq!(w.context_tokens(), u32::MAX);
    }

    #[test]
    fn the_sum_with_the_base_saturates_at_the_top_of_the_range() {
        let w = AgentWorkload {
            turns: 1,
            turn_tokens: u32::MAX,
        };
        assert_eq!(w.context_tokens(), u32::MAX);
    }

    #[test]
    fn describe_names_both_assumptions_the_base_and_the_result() {
        let w = AgentWorkload {
            turns: 10,
            turn_tokens: 2_048,
        };
        let text = w.describe();
        assert!(text.contains("10 tool turns"), "{text}");
        assert!(text.contains("2048 tokens"), "{text}");
        assert!(text.contains(&DEFAULT_ESTIMATION_CTX.to_string()), "{text}");
        assert!(text.contains(&w.context_tokens().to_string()), "{text}");
    }
}
