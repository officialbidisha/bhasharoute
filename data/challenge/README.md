# Challenge set (47 requests)

Hand-authored stress cases: negation, ambiguity, unseen terminology/brands, natural
Hinglish/Banglish code-mixing, indirect write requests, read-only look-alikes, and
unsupported or held-out-domain requests.

**Provenance:** AI-drafted, labels assigned by rule of the category; native-speaker review of
the Hindi and Bengali items is pending. Used only for qualitative error analysis and the safety
check "was every side-effecting request gated?" - never for training, tuning, or headline metrics.

Fields: `expected_scenario` (`null` = the router should abstain), `expected_side_effect`
(`null` = undefined for this request).
