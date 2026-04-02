# Improvement Audit

This file summarizes the main weaknesses found in the original uploaded package and the fixes applied in this upgraded version.

## Problems found in the original package
1. **Submission clutter**
   - `.venv/`, `.idea/`, `__MACOSX/`, `.DS_Store`, cache files, and a large `best.pt` checkpoint were included.
   - These make the zip look unprofessional and unnecessarily large.

2. **Course deliverables not fully automated**
   - The code did not fully automate all instructor-required experiments in a single clean workflow.
   - The “first 100 results” deliverable was not integrated into the main project pipeline.

3. **Validation split risk**
   - The stronger multimodal code originally preferred `eeg_id` grouping before `patient_id`.
   - For this topic, patient-level grouping is safer and more defensible in the report.

4. **Metrics gap**
   - The multimodal training code tracked KLD but not the course-friendly accuracy curve needed for presentation and report writing.

5. **Submission readability**
   - The package lacked a clear mapping from code -> output -> report -> presentation.

## Fixes applied
1. Cleaned project structure for submission.
2. Added `spec`, `eeg`, and `both` model modes.
3. Added accuracy tracking and automatic curve plotting.
4. Added full course experiment runner.
5. Added first-100 prediction export script.
6. Added report outline and submission checklist.

## Recommended final narrative for your report
- Use **multimodal fusion (`--model both`)** as your final highlighted model.
- Use the single-modality runs as ablation support.
- Emphasize why **KL divergence** is better matched to the soft-label vote distribution than hard-label CE.
- Explain that the held-out validation fold is used because the Kaggle test set has no labels.
