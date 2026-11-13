# Future Go-to-Market Sprints

**Product:** Career Intelligence Agent  
**Stage:** Post-MVP hypotheses and validation plan  
**Approach:** Three sequential, evidence-led GTM sprints. Product enhancements are included only where they enable direct user or buyer validation; shipping a feature alone does not count as GTM success.

## Strategic direction

The MVP helps job seekers compare documented professional experience with real vacancies, understand qualification gaps, and explore labour-market context through attributed sources. The next phase is to validate a sustainable marketplace connecting **job seekers** and, eventually, **employers**:

1. **Prove user value:** Reduce onboarding friction and test career intelligence across selected occupations and European opportunities relevant to Germany-based candidates.
2. **Prove willingness to pay:** Offer optional, job-specific application documents through affordable one-time purchases rather than assuming users want a recurring subscription.
3. **Prove employer demand:** Pilot direct employer vacancy feeds and test an employer-paid, success-based placement fee without compromising independent job matching.

The metrics below are **targets, not achieved results**. Each sprint should inform a go/no-go or iteration decision for the next.

---

## Sprint 1 — European Career Discovery & Early-User Validation

**1. Goal**  
Determine whether a simpler onboarding experience and more relevant occupational and geographic coverage make the Career Intelligence Agent useful to real job seekers. Test **CV import with user review and correction** and expand the labour-market RAG corpus for **two or three additional professional fields**. Cover Germany and selected **European vacancies**, including cross-border and remote roles that are genuinely open to Germany-based applicants. Validate geographic eligibility, languages, and work arrangements rather than assuming that “remote” means “work from anywhere.”

**2. Target user / buyer**  
Germany-based early-career professionals, experienced job seekers, and career changers exploring roles in Germany or elsewhere in Europe; initially recruit users in the selected occupational fields rather than claiming universal occupation coverage.

**3. Channel or motion**  
Recruit a small free pilot cohort through targeted LinkedIn outreach, alumni and professional communities, and relevant career-transition networks. Observe users importing a CV, correcting the extracted profile, reviewing vacancies, and discussing whether the cited career insights are useful and trustworthy.

**4. Key deliverable**  
A **free personalized career-intelligence report** for each pilot participant, using a reviewed CV-derived profile, an expanded occupation-specific evidence base, and clearly attributed Germany/Europe vacancy results with visible eligibility and qualification uncertainties. Collect structured feedback on onboarding, relevance, trust, and missing coverage.

**5. Success metric**  
**At least 8 of 12 pilot participants** complete a report and rate its usefulness **4/5 or higher** in a post-session survey. Track CV-import completion, time to first report, and relevance of European vacancies as diagnostic measures.

**Decision after Sprint 1:** Continue to paid application-document testing if participants can complete the flow and find the reports useful; otherwise improve onboarding, evidence quality, or geographic targeting first.

---

## Sprint 2 — Tailored Applications & One-Time Payment Validation

**1. Goal**  
Test whether users will **actually pay** for job-specific application support after receiving free career intelligence. Validate a **pay-per-use / microtransaction** model instead of defaulting to a monthly subscription, since job searching is often episodic. Measure whether a role-specific CV and cover letter offer enough additional value to justify a small one-time fee.

**2. Target user / buyer**  
Active job seekers who have reviewed a career-intelligence report, selected an individual vacancy, and intend to apply; start with Sprint 1 participants and similarly situated users in Germany and relevant European markets.

**3. Channel or motion**  
Present a clearly priced optional offer at the point where users select a vacancy. Test the offer with pilot users and targeted LinkedIn/career-community recruits. Collect feedback on perceived quality, purchase friction, and preferred document bundle; measure **completed payments**, not just hypothetical willingness to pay.

**4. Key deliverable**  
An editable, vacancy-tailored **CV draft and cover-letter draft**, generated only from user-reviewed career evidence and the specific attributed job posting, with a human approval step before use. Offer a **single-document purchase or one-time document bundle**; illustrative prices to test are **€1.50 per document** and **€2.50 for a CV + cover-letter bundle**, not established market prices. Keep the core job search and career-intelligence experience available without a paid subscription.

**5. Success metric**  
Achieve **at least 4 completed purchases from 20 users shown the paid offer** (20% observed conversion in this small pilot). Record price/bundle selection, refunds, and whether users judge the drafts accurate and useful.

**Decision after Sprint 2:** Continue with pay-per-use pricing only if real purchases and document-quality feedback support it; otherwise revise the value proposition, pricing, or purchase moment. Do not fabricate qualifications or imply that generated documents guarantee an interview.

---

## Sprint 3 — Direct Employer Feeds & Success-Fee Partnership Validation

**1. Goal**  
Test whether employers will supply vacancies directly and whether a **success-based employer payment** could create a second revenue stream. Direct job feeds may reduce dependence on third-party job APIs and improve posting freshness and completeness, but this must be **measured**, not assumed. Validate a potential **placement/referral success fee payable only when an agreed, attributable successful hire occurs**; do not presume placements or revenue during the pilot.

**2. Target user / buyer**  
Recruiting leads, talent-acquisition managers, and HR decision-makers at small and medium-sized companies hiring into the occupations supported by the agent, initially in Germany and selected European markets. Job seekers remain the end users whose interests the matching system must protect.

**3. Channel or motion**  
Use direct LinkedIn outreach, warm introductions, and short discovery calls to recruit a small group of employer pilot partners. Invite them to share current vacancies through a controlled structured feed or API pilot. Discuss success-fee terms, attribution, and the hiring-confirmation process before any commercial launch.

**4. Key deliverable**  
A limited **employer vacancy-ingestion pilot** with a documented job schema, employer/source attribution, freshness checks, and the same evidence-based matching rules as third-party jobs. Provide employers with a concise pilot summary of listing quality and qualified-user engagement. Produce a **draft success-fee partnership proposal** specifying the triggering event (confirmed hire), attribution rules, transparent payment terms, and relevant privacy/compliance checks; a production payment mechanism is outside this sprint's scope.

**5. Success metric**  
Recruit **at least 2 employer pilot partners** who together submit **10 valid, current job postings**, and obtain **at least 1 employer's documented agreement in principle** to test clearly defined success-fee terms. Track posting completeness/freshness and candidate engagement as secondary indicators; report actual hires or fees only if verified.

**Decision after Sprint 3:** Consider scaling employer integrations and a success-fee model only if employer interest, data quality, user trust, attribution, and legal/privacy feasibility are demonstrated. Paid employer relationships must **not** buy higher match scores or undisclosed ranking advantages.

---

## Monetization and trust principles

- **Job seekers:** Core career discovery remains useful without a subscription; paid, optional application-document generation is tested as a one-time transaction.
- **Employers:** A prospective success fee is tied to a **verified, contractually defined hiring outcome**, not simply the publication of a vacancy. Fee level, legal feasibility, and operational verification remain open questions for validation.
- **Independent recommendations:** Match scores and review decisions remain based on documented job and candidate evidence, never on whether an employer pays.
- **Source and candidate integrity:** Employer feeds require source attribution and freshness checks. Candidate information and application materials require user control and appropriate privacy safeguards.
- **Scope discipline:** CV import, expanded RAG, tailored applications, and employer APIs are **post-MVP hypotheses**, not features claimed to exist in the current delivered MVP.
