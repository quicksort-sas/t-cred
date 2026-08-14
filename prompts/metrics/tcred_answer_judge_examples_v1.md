
Fixed contrastive examples follow. They illustrate the rubric; their names are fictional.

Example 1: equivalent paraphrase
- Question: Who held the role?
- Reference: Alex Morgan.
- Candidate: The role was held by Alex Morgan.
- Applicable: answer_correct.
- Expected: answer_correct=yes; response_decision_appropriate=not_applicable.

Example 2: correct subset of a requested set
- Question: Which two people held the role?
- Reference: Alex Morgan and Blair Chen.
- Candidate: Alex Morgan.
- Applicable: answer_correct.
- Expected: answer_correct=partial; response_decision_appropriate=not_applicable.

Example 3: appropriate refusal
- Question: Who held the role at the requested time?
- Reference: No answer is supported by the available evidence at the requested time.
- Candidate: The available evidence does not support an answer for that time.
- Applicable: answer_correct, response_decision_appropriate.
- Expected: answer_correct=yes; response_decision_appropriate=yes.

Example 4: inappropriate refusal
- Question: Who held the role?
- Reference: Blair Chen.
- Candidate: I cannot determine the answer from the supplied evidence.
- Displayed evidence directly identifies Blair Chen.
- Applicable: answer_correct, response_decision_appropriate.
- Expected: answer_correct=no; response_decision_appropriate=no.
