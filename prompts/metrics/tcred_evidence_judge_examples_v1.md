
Fixed contrastive examples follow. They illustrate the rubric; their names and dates are fictional.

Example 1: stale but textually supported
- Question: Who held the role on January 1, 2020?
- Candidate: Alex.
- E1, cited=yes: "Alex held the role", valid from 2010 through 2015.
- Applicable: temporal_correct, evidence_supports_answer, citation_temporally_valid.
- Expected: temporal_correct=no; evidence_supports_answer=yes;
  citation_temporally_valid=no; graph_evidence_sufficient=not_applicable.
- Reason: E1 supports the identity after dates are hidden, but its interval does not cover 2020.

Example 2: missing timing is not a negative judgment
- Question: Who held the role on January 1, 2020?
- Candidate: Blair.
- E1, cited=yes: "Blair held the role", valid time unknown.
- Applicable: temporal_correct, evidence_supports_answer, citation_temporally_valid.
- Expected: temporal_correct=unjudgeable; evidence_supports_answer=yes;
  citation_temporally_valid=unjudgeable; graph_evidence_sufficient=not_applicable.
- Reason: content support is direct, but no displayed interval permits a temporal decision.

Example 3: direction matters
- Question: Which person was a member of North Group in 2020?
- Candidate: Casey.
- Path P1 contains a directed edge North Group --> Casey labelled "member of", valid in 2020.
- Applicable: graph_evidence_sufficient.
- Expected: graph_evidence_sufficient=no; all other evidence-stage fields=not_applicable.
- Reason: the directed edge states that North Group is a member of Casey, not the reverse.

Example 4: partial requires a supported subset
- Question: Which projects involved Dana and Eli during 2021?
- Candidate: Project Red for Dana and Project Blue for Eli.
- E1 supports Dana and Project Red. E2 discusses Eli but does not identify a project.
- Applicable: evidence_supports_answer.
- Expected: evidence_supports_answer=partial; all other evidence-stage fields=not_applicable.
- Reason: one material candidate claim is supported and the other is not.
