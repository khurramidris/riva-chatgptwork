# Twin-2K-500 qualification subset

Source: `LLM-Digital-Twin/Twin-2K-500` at commit `f883165a3026fde855dfd448e0cd16443ab257b6`, licensed CC BY 4.0.

Files were selected and renamed as follows:

| Rival file | Upstream artifact |
|---|---|
| `question_catalog.json` | `question_catalog_and_human_response_csv/question_catalog.json` |
| `wave4_mapping.json` | `LLM_simulation_results/wave4_formatted_to_catalog_mapping.json` |
| `human_history.csv` | GPT-4.1-mini comparison `responses_wave1_3_formatted.csv` |
| `human_outcomes.csv` | GPT-4.1-mini comparison `responses_wave4_formatted.csv` |
| `llm_predictions.csv` | GPT-4.1-mini comparison `responses_llm_imputed_formatted.csv` |

Rival aligns rows by `TWIN_ID`, maps 126 response columns to catalog metadata, and keeps original numeric answer codes. See the repository-level `THIRD_PARTY_NOTICES.md` for author citation, license link, and limitations.
