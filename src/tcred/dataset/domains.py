from __future__ import annotations

from dataclasses import dataclass

from tcred.dataset.models import EntityType, Relation


@dataclass(frozen=True)
class DomainSpec:
    domain: str
    blueprint: str
    context_type: EntityType
    answer_type: EntityType
    object_type: EntityType
    relation: Relation
    object_names: tuple[str, ...]
    context_nouns: tuple[str, ...]
    source_type: str
    question_noun: str
    path_relation: Relation = Relation.AFFILIATED_WITH


DOMAIN_SPECS: tuple[DomainSpec, ...] = (
    DomainSpec(
        domain="corporate_roles",
        blueprint="role_succession",
        context_type=EntityType.ORGANIZATION,
        answer_type=EntityType.PERSON,
        object_type=EntityType.ROLE,
        relation=Relation.HELD_ROLE,
        object_names=("Director", "Chief Operations Officer", "Research Lead", "Compliance Lead"),
        context_nouns=("Labs", "Systems", "Holdings", "Analytics"),
        source_type="registry_record",
        question_noun="role holder",
    ),
    DomainSpec(
        domain="sports_memberships",
        blueprint="membership_transfer",
        context_type=EntityType.TEAM,
        answer_type=EntityType.PERSON,
        object_type=EntityType.ROLE,
        relation=Relation.MEMBER_OF,
        object_names=("starting squad member", "captain", "loan player", "reserve player"),
        context_nouns=("FC", "United", "Athletic", "Rovers"),
        source_type="team_roster",
        question_noun="team member",
    ),
    DomainSpec(
        domain="policies",
        blueprint="policy_supersession",
        context_type=EntityType.ORGANIZATION,
        answer_type=EntityType.POLICY,
        object_type=EntityType.ROLE,
        relation=Relation.POLICY_EFFECTIVE,
        object_names=("data-retention rule", "access-control rule", "appeals procedure"),
        context_nouns=("Authority", "Agency", "Commission", "Office"),
        source_type="policy_register",
        question_noun="effective policy",
    ),
    DomainSpec(
        domain="contracts",
        blueprint="contract_renewal",
        context_type=EntityType.ORGANIZATION,
        answer_type=EntityType.CONTRACT,
        object_type=EntityType.ROLE,
        relation=Relation.CONTRACT_ACTIVE,
        object_names=("service agreement", "licensing agreement", "support contract"),
        context_nouns=("Partners", "Services", "Networks", "Consulting"),
        source_type="contract_register",
        question_noun="active contract",
    ),
    DomainSpec(
        domain="product_versions",
        blueprint="version_lifecycle",
        context_type=EntityType.PRODUCT,
        answer_type=EntityType.PRODUCT_VERSION,
        object_type=EntityType.ROLE,
        relation=Relation.SUPPORT_WINDOW,
        object_names=("supported release", "stable release", "security-supported release"),
        context_nouns=("Suite", "Engine", "Platform", "Cloud"),
        source_type="release_notes",
        question_noun="supported version",
    ),
    DomainSpec(
        domain="event_timelines",
        blueprint="event_chain",
        context_type=EntityType.PROJECT,
        answer_type=EntityType.EVENT,
        object_type=EntityType.ROLE,
        relation=Relation.EVENT_OCCURS,
        object_names=("milestone", "review", "deployment", "audit"),
        context_nouns=("Project", "Program", "Initiative", "Rollout"),
        source_type="timeline_record",
        question_noun="event",
        path_relation=Relation.EVENT_PRECEDES,
    ),
    DomainSpec(
        domain="research_projects",
        blueprint="project_participation",
        context_type=EntityType.PROJECT,
        answer_type=EntityType.PERSON,
        object_type=EntityType.ROLE,
        relation=Relation.PROJECT_PARTICIPANT,
        object_names=("principal investigator", "coordinator", "data steward"),
        context_nouns=("Study", "Consortium", "Trial", "Lab"),
        source_type="project_roster",
        question_noun="project participant",
    ),
    DomainSpec(
        domain="locations",
        blueprint="facility_location",
        context_type=EntityType.ORGANIZATION,
        answer_type=EntityType.LOCATION,
        object_type=EntityType.ROLE,
        relation=Relation.LOCATED_AT,
        object_names=("headquarters", "archive site", "operations center"),
        context_nouns=("Institute", "Archive", "Center", "Bureau"),
        source_type="facility_register",
        question_noun="location",
    ),
)
