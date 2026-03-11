# Email to Goodfire — Feature Labels Request

**To:** contact@goodfire.ai
**Subject:** Request for autointerp feature labels — Llama 3.3 70B SAE (layer 50)

---

Hi Goodfire team,

I'm Jord Nguyen, an AI safety researcher (papers at ICLR/AAAI, currently at a Constellation research fellowship in Berkeley). I'm working on a replication and extension of Berg et al. (2025) "Large Language Models Report Subjective Experience Under Self-Referential Processing," which used the Goodfire Ember API to steer deception-related SAE features in Llama 3.3 70B and measure effects on consciousness self-reports.

Since the Ember API has been deprecated, I've been self-hosting your open-source SAE weights from HuggingFace (Goodfire/Llama-3.3-70B-Instruct-SAE-l50). The weights work well and I've been able to do contrastive feature discovery and decoder-direction steering. However, the feature labels that were previously accessible through the API would significantly strengthen the research — both for identifying the specific features the paper used and for validating our contrastive discovery approach.

**Would it be possible to share the autointerp feature labels for the Llama 3.3 70B Instruct SAE (layer 50)?** A CSV or JSON mapping feature indices to labels would be ideal — similar to the format you published for the DeepSeek R1 SAEs in the r1-interpretability repo.

The research is non-commercial and focused on understanding how SAE features relate to model honesty and self-report behavior. Happy to share our findings and cite Goodfire's work.

Thanks for considering this — and for open-sourcing the SAE weights in the first place.

Best,
Jord Nguyen
