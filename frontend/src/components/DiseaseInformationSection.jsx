import React, { useState } from "react";

export function DiseaseInformationSection({ diseaseInfo, overallPrediction, screening }) {
  const [expandAll, setExpandAll] = useState(false);

  const candidateCount = screening?.candidate_count ?? 0;
  const noCandidatesFound = candidateCount === 0;

  if (!diseaseInfo) {
    if (noCandidatesFound) {
      return (
        <section className="results-section section-disease-info" aria-labelledby="disease-info-heading">
          <div className="section-header-row">
            <div>
              <span className="section-tag">Agricultural Knowledge Base</span>
              <h2 id="disease-info-heading" className="section-title">Screening Outcome: No Candidate Regions Identified</h2>
            </div>
            <div className="disease-info-header-actions">
              <span className="status-pill badge-neutral">No Candidates ≥ 60% Threshold</span>
            </div>
          </div>

          <div className="disease-context-banner banner-neutral" role="status">
            <p className="context-banner-lead">
              <strong>Field Assessment Context: </strong>
              No disease-specific profile was selected because no localized candidate crop crossed the 60% screening threshold in whole-plant mode.
            </p>
            <p className="context-banner-sub">
              Use the field scouting checklist below to inspect early-stage foliar symptoms, lower leaf surfaces, or non-foliar stress factors that image crops may miss.
            </p>
          </div>

          <div className="disease-disclaimer-card" role="note">
            <div className="disclaimer-inner">
              <span className="disclaimer-badge">Mandatory Educational Disclaimer</span>
              <p className="disclaimer-text">
                This information is provided for educational purposes. Model predictions are not confirmed diagnoses. Disease symptoms can resemble those of other conditions. Consult a qualified agricultural expert or local agricultural extension service before making treatment decisions.
              </p>
            </div>
          </div>

          <div className="disease-subsections-grid">
            <details className="disease-sub-accordion" open={true}>
              <summary className="accordion-summary">
                <span className="summary-title">Recommended Next Steps for Field Assessment</span>
                <span className="summary-tag">Field Best Practices</span>
              </summary>
              <div className="accordion-content">
                <ul className="next-steps-list">
                  <li><strong>Rescan with closer framing:</strong> Reposition camera closer to individual leaves or foliage of concern, ensuring sharp focus and diffused natural lighting.</li>
                  <li><strong>Check undersides of leaves and stems:</strong> Inspect lower leaf surfaces, petioles, and stem nodes for hidden spore pustules, fungal mycelia, or small insect vectors.</li>
                  <li><strong>Monitor for early-stage symptom development:</strong> Recheck the crop in 24–48 hours for expanding chlorotic margins, water-soaked flecks, or wilting.</li>
                  <li><strong>Consult an extension specialist if symptoms persist:</strong> Contact your local cooperative extension office or university plant disease diagnostic clinic for physical laboratory verification.</li>
                </ul>
              </div>
            </details>
          </div>
        </section>
      );
    }
    return null;
  }

  // Determine state context
  const isUncertain = Boolean(
    overallPrediction?.is_uncertain ||
    overallPrediction?.needs_review ||
    (screening?.candidate_regions?.[0]?.needs_review && screening?.mode === "whole_plant")
  );

  const isHealthy = Boolean(
    diseaseInfo.is_healthy ||
    overallPrediction?.is_healthy
  );

  // Context-aware heading and subtitle
  let headingTitle = "Disease Information";
  let statusBadge = "Educational Reference · Predicted Disease Class";
  let badgeTone = "badge-info";
  let contextExplanation = "";

  if (isUncertain) {
    headingTitle = "About the Tentative Prediction";
    statusBadge = "Tentative Hypothesis · Score Below 60% Review Threshold";
    badgeTone = "badge-uncertain";
    contextExplanation = (
      <div className="disease-context-banner banner-uncertain" role="status">
        <p className="context-banner-lead">
          <strong>Tentative Prediction Context: </strong>
          The model score is below the operational review threshold (60%). The educational reference below describes the tentative match (<strong>{diseaseInfo.overview?.common_name || diseaseInfo.condition}</strong>).
        </p>
        <p className="context-banner-sub">
          This educational information describes the biology of this condition and does <em>not</em> confirm that your plant has this disease. Consult a qualified agricultural diagnostic lab or extension educator for professional confirmation.
        </p>
      </div>
    );
  } else if (isHealthy) {
    headingTitle = "Plant Health & Crop Care Information";
    statusBadge = "Educational Reference · Healthy Foliage Profile";
    badgeTone = "badge-healthy";
    contextExplanation = (
      <div className="disease-context-banner banner-healthy" role="status">
        <p className="context-banner-lead">
          <strong>Foliar Health Context: </strong>
          The classifier did not identify visible foliar lesions matching its modeled disease classes for <strong>{diseaseInfo.crop}</strong>.
        </p>
        <p className="context-banner-sub">
          The reference guide below outlines normal foliar characteristics and sound agronomic hygiene to sustain crop vitality. <em>Note:</em> A healthy screening output indicates the absence of modeled foliar spots; it does not preclude latent viral infections, root rot, or unindexed nutrient disorders.
        </p>
      </div>
    );
  } else {
    headingTitle = "Disease Information";
    statusBadge = "Educational Reference · High-Confidence Hypothesis";
    badgeTone = "badge-confident";
    contextExplanation = (
      <div className="disease-context-banner banner-confident" role="status">
        <p className="context-banner-lead">
          <strong>Model Hypothesis Context: </strong>
          The educational reference below describes the predicted disease class (<strong>{diseaseInfo.overview?.common_name || diseaseInfo.condition}</strong>).
        </p>
        <p className="context-banner-sub">
          Model predictions are automated computer vision screening outputs, not laboratory-certified clinical diagnoses. Use this profile to understand pathogen biology, field symptoms, and extension-recommended management practices.
        </p>
      </div>
    );
  }

  return (
    <section className="results-section section-disease-info" aria-labelledby="disease-info-heading">
      {/* Header Row */}
      <div className="section-header-row">
        <div>
          <span className="section-tag">Agricultural Knowledge Base</span>
          <h2 id="disease-info-heading" className="section-title">{headingTitle}</h2>
        </div>
        <div className="disease-info-header-actions">
          <span className={`status-pill ${badgeTone}`}>{statusBadge}</span>
          <button
            type="button"
            className="toggle-expand-button"
            onClick={() => setExpandAll((prev) => !prev)}
            aria-label={expandAll ? "Collapse optional subsections" : "Expand all subsections"}
          >
            {expandAll ? "Collapse Subsections" : "Expand All"}
          </button>
        </div>
      </div>

      {/* Context-aware Explanation Banner */}
      {contextExplanation}

      {/* Mandatory Educational Disclaimer */}
      <div className="disease-disclaimer-card" role="note">
        <div className="disclaimer-inner">
          <span className="disclaimer-badge">Mandatory Educational Disclaimer</span>
          <p className="disclaimer-text">
            This information is provided for educational purposes. Model predictions are not confirmed diagnoses. Disease symptoms can resemble those of other conditions. Consult a qualified agricultural expert or local agricultural extension service before making treatment decisions.
          </p>
        </div>
      </div>

      {/* Structured Subsections */}
      <div className="disease-subsections-grid">
        {/* SUBSECTION 1: OVERVIEW */}
        <details className="disease-sub-accordion" open={true}>
          <summary className="accordion-summary">
            <span className="summary-title">1. Disease Overview</span>
            <span className="summary-tag">{diseaseInfo.overview?.pathogen_type || "Pathology Overview"}</span>
          </summary>
          <div className="accordion-content">
            <div className="overview-meta-grid">
              <div className="meta-box">
                <span className="meta-field-label">Common Name</span>
                <strong className="meta-field-value">{diseaseInfo.overview?.common_name || diseaseInfo.condition}</strong>
              </div>
              <div className="meta-box">
                <span className="meta-field-label">Scientific Name</span>
                <em className="meta-field-value italic">{diseaseInfo.overview?.scientific_name || "N/A"}</em>
              </div>
              <div className="meta-box">
                <span className="meta-field-label">Pathogen Type</span>
                <span className="pathogen-pill">{diseaseInfo.overview?.pathogen_type || "Unclassified"}</span>
              </div>
              <div className="meta-box">
                <span className="meta-field-label">Target Crop</span>
                <strong className="meta-field-value">{diseaseInfo.crop}</strong>
              </div>
            </div>
            <div className="overview-narrative">
              <p>{diseaseInfo.overview?.description}</p>
            </div>
          </div>
        </details>

        {/* SUBSECTION 2: SYMPTOMS */}
        <details className="disease-sub-accordion" open={true}>
          <summary className="accordion-summary">
            <span className="summary-title">2. Visible Foliar Symptoms</span>
            <span className="summary-tag">Field Diagnosis Guide</span>
          </summary>
          <div className="accordion-content">
            <div className="info-block-row">
              <strong className="info-row-label">Typical Leaf Symptoms:</strong>
              <p className="info-row-text">{diseaseInfo.symptoms?.leaf_symptoms}</p>
            </div>
            <div className="info-block-row">
              <strong className="info-row-label">Lesion Appearance & Morphology:</strong>
              <p className="info-row-text">{diseaseInfo.symptoms?.lesion_appearance}</p>
            </div>
            <div className="info-block-row">
              <strong className="info-row-label">Symptom Progression:</strong>
              <p className="info-row-text">{diseaseInfo.symptoms?.progression}</p>
            </div>
            <div className="info-block-row">
              <strong className="info-row-label">Other Plant Organs Affected:</strong>
              <p className="info-row-text">{diseaseInfo.symptoms?.other_parts_affected}</p>
            </div>
          </div>
        </details>

        {/* SUBSECTION 3: CAUSES AND SPREAD */}
        <details className="disease-sub-accordion" open={expandAll}>
          <summary className="accordion-summary">
            <span className="summary-title">3. Causes and Epidemiological Spread</span>
            <span className="summary-tag">Transmission Factors</span>
          </summary>
          <div className="accordion-content">
            <div className="info-block-row">
              <strong className="info-row-label">Causal Organism:</strong>
              <p className="info-row-text"><em>{diseaseInfo.causes_and_spread?.causal_organism}</em></p>
            </div>
            <div className="info-block-row">
              <strong className="info-row-label">Favorable Environmental Conditions:</strong>
              <p className="info-row-text">{diseaseInfo.causes_and_spread?.favorable_conditions}</p>
            </div>
            <div className="info-block-row">
              <strong className="info-row-label">Modes of Transmission & Dispersal:</strong>
              <p className="info-row-text">{diseaseInfo.causes_and_spread?.transmission_modes}</p>
            </div>
            <div className="info-block-row">
              <strong className="info-row-label">Host Susceptibility Factors:</strong>
              <p className="info-row-text">{diseaseInfo.causes_and_spread?.susceptibility_factors}</p>
            </div>
          </div>
        </details>

        {/* SUBSECTION 4: POSSIBLE MANAGEMENT */}
        <details className="disease-sub-accordion" open={expandAll}>
          <summary className="accordion-summary">
            <span className="summary-title">4. Possible Management Practices</span>
            <span className="summary-tag">Agronomic & Extension Advice</span>
          </summary>
          <div className="accordion-content">
            <div className="info-block-row">
              <strong className="info-row-label">Cultural & Agronomic Practices:</strong>
              <p className="info-row-text">{diseaseInfo.management?.cultural_practices}</p>
            </div>
            <div className="info-block-row">
              <strong className="info-row-label">Sanitation & Debris Management:</strong>
              <p className="info-row-text">{diseaseInfo.management?.sanitation}</p>
            </div>
            <div className="info-block-row">
              <strong className="info-row-label">Environmental Canopy Control:</strong>
              <p className="info-row-text">{diseaseInfo.management?.environmental_control}</p>
            </div>
            <div className="info-block-row professional-alert-row">
              <strong className="info-row-label">Professional Extension Consultation:</strong>
              <p className="info-row-text font-medium">{diseaseInfo.management?.professional_advice}</p>
            </div>
            <div className="info-block-row treatment-options-row">
              <strong className="info-row-label">General Treatment Options (Extension Supported):</strong>
              <p className="info-row-text">{diseaseInfo.management?.general_treatments}</p>
              <span className="treatment-disclaimer-note">
                Always follow product labels, local pesticide regulations, and pre-harvest intervals (PHI). Consult your local county extension office for current legal chemistries.
              </span>
            </div>
          </div>
        </details>

        {/* SUBSECTION 5: PREVENTION */}
        <details className="disease-sub-accordion" open={expandAll}>
          <summary className="accordion-summary">
            <span className="summary-title">5. Preventative Hygiene & Cultivars</span>
            <span className="summary-tag">Long-Term Defense</span>
          </summary>
          <div className="accordion-content">
            <div className="info-block-row">
              <strong className="info-row-label">Field & Crop Hygiene:</strong>
              <p className="info-row-text">{diseaseInfo.prevention?.crop_hygiene}</p>
            </div>
            <div className="info-block-row">
              <strong className="info-row-label">Irrigation Best Practices:</strong>
              <p className="info-row-text">{diseaseInfo.prevention?.irrigation_practices}</p>
            </div>
            <div className="info-block-row">
              <strong className="info-row-label">Scouting & Early Detection:</strong>
              <p className="info-row-text">{diseaseInfo.prevention?.monitoring_and_scouting}</p>
            </div>
            <div className="info-block-row">
              <strong className="info-row-label">Resistant Varieties / Cultivars:</strong>
              <p className="info-row-text">{diseaseInfo.prevention?.resistant_varieties}</p>
            </div>
          </div>
        </details>

        {/* SUBSECTION 6: INSTITUTIONAL EXTENSION REFERENCES */}
        <details className="disease-sub-accordion" open={expandAll}>
          <summary className="accordion-summary">
            <span className="summary-title">6. Authoritative Extension References</span>
            <span className="summary-tag">USDA / University Extension</span>
          </summary>
          <div className="accordion-content">
            <p className="reference-intro-note">
              The descriptions, symptom profiles, and management protocols above are synthesized from published extension guides by land-grant universities and agricultural research agencies:
            </p>
            <div className="extension-reference-list">
              {diseaseInfo.references?.map((ref, idx) => (
                <div className="extension-ref-card" key={`ref-${idx}-${ref.institution}`}>
                  <div className="ref-institution-badge">{ref.institution}</div>
                  <strong className="ref-title">{ref.title}</strong>
                  <a
                    className="ref-link"
                    href={ref.url}
                    target="_blank"
                    rel="noreferrer"
                    aria-label={`Open ${ref.title} from ${ref.institution}`}
                  >
                    View Official Extension Resource →
                  </a>
                </div>
              ))}
            </div>
          </div>
        </details>
      </div>
    </section>
  );
}
