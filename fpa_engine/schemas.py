from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_.\-]*$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceKind(str, Enum):
    sec_filing = "sec_filing"
    sec_exhibit = "sec_exhibit"
    ir_release_pdf = "ir_release_pdf"
    call_transcript = "call_transcript"


class VerificationMethod(str, Enum):
    document_read = "document_read"
    edgar_fts_phrase_match = "edgar_fts_phrase_match"
    transcript = "transcript"


class Origin(str, Enum):
    sourced = "sourced"
    derived = "derived"


class DisclosureStatus(str, Enum):
    disclosed = "disclosed"
    not_disclosed = "not_disclosed"
    not_collected = "not_collected"
    back_solved = "back_solved"


class Unit(str, Enum):
    usd_m = "usd_m"
    pct = "pct"
    pts = "pts"
    bps = "bps"


class PeriodType(str, Enum):
    three_months = "three_months"
    ytd = "ytd"
    rolling_3m = "rolling_3m"


class PriceBasis(str, Enum):
    current = "current"
    price_neutral = "price_neutral"
    mixed = "mixed"
    not_applicable = "not_applicable"


class EvidenceType(str, Enum):
    explicit = "explicit"
    interpretive = "interpretive"


class ReviewStatus(str, Enum):
    validated = "validated"
    needs_review = "needs_review"
    rejected = "rejected"


class BridgeType(str, Enum):
    company_disclosed_bridge = "company_disclosed_bridge"
    accounting_identity = "accounting_identity"
    non_additive_set = "non_additive_set"


class ReconciliationStatus(str, Enum):
    RECONCILED = "RECONCILED"
    UNRESOLVED = "UNRESOLVED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CalcStatus(str, Enum):
    computed = "computed"
    unavailable = "unavailable"
    not_computable = "not_computable"


class SourceReference(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    source_kind: SourceKind
    publisher: str
    document: str
    filing_type: str | None = None
    filing_date: date | None = None
    vintage_date: date | None = None
    accession_number: str | None = Field(default=None, pattern=r"^\d{10}-\d{2}-\d{6}$")
    section: str | None = None
    locator: str | None = None
    excerpt: str | None = None
    url: str
    verification_method: VerificationMethod
    retrieved_on: date | None = None


class FinancialFact(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    metric: str
    value: Decimal | None = None
    unit: Unit
    period: str
    origin: Origin
    disclosure_status: DisclosureStatus
    source_ids: list[str] = Field(default_factory=list)
    formula: str | None = None
    inputs: list[str] = Field(default_factory=list)
    basis: str
    comparability_notes: list[str] = Field(default_factory=list)
    precision: Decimal | None = None
    scope: str
    period_type: PeriodType
    vintage_source_id: str | None = None
    price_basis: PriceBasis
    sign_convention: str
    approximate: bool = False
    review_flags: list[str] = Field(default_factory=list)
    benchmark_cell: str | None = None

    @model_validator(mode="after")
    def _shape_matches_origin(self) -> FinancialFact:
        if self.origin is Origin.derived:
            if not self.formula or not self.inputs:
                raise ValueError(f"{self.id}: derived facts need a registered formula and inputs")
        else:
            if self.formula or self.inputs:
                raise ValueError(f"{self.id}: sourced facts cannot declare a formula or inputs")
            if self.value is None:
                raise ValueError(f"{self.id}: sourced facts need a value (record gaps as DisclosureGap)")
            if self.precision is None or self.precision <= 0:
                raise ValueError(f"{self.id}: sourced facts need a positive disclosure precision")
        return self


class DisclosureGap(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    metric: str
    scope: str
    period: str
    disclosure_status: DisclosureStatus
    reason: str
    evidence_source_ids: list[str] = Field(default_factory=list)
    benchmark_cell: str | None = None

    @model_validator(mode="after")
    def _gap_has_no_value_status(self) -> DisclosureGap:
        if self.disclosure_status not in (DisclosureStatus.not_disclosed, DisclosureStatus.not_collected):
            raise ValueError(f"{self.id}: a gap must be not_disclosed or not_collected")
        return self


class NarrativeEvidence(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    topic: str
    claim: str
    evidence_type: EvidenceType
    verbatim_quote: str | None = None
    speaker: str | None = None
    source_kind: SourceKind
    scope: str
    source_ids: list[str] = Field(default_factory=list)
    period: str
    related_fact_ids: list[str] = Field(default_factory=list)
    premise_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    counter_evidence_ids: list[str] = Field(default_factory=list)
    review_status: ReviewStatus

    @model_validator(mode="after")
    def _evidence_shape(self) -> NarrativeEvidence:
        if self.evidence_type is EvidenceType.explicit:
            if self.review_status is ReviewStatus.validated and not self.verbatim_quote:
                raise ValueError(f"{self.id}: validated explicit evidence needs a verbatim quote")
            if self.assumptions:
                raise ValueError(f"{self.id}: explicit evidence cannot rest on assumptions")
        if self.evidence_type is EvidenceType.interpretive and not self.premise_ids:
            raise ValueError(f"{self.id}: interpretive evidence must cite its premises")
        return self


class BridgeDefinition(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    name: str
    period: str
    scope: str
    bridge_type: BridgeType
    starting_fact: str | None = None
    ending_fact: str | None = None
    components: dict[str, str] = Field(default_factory=dict)
    unquantified_components: list[str] = Field(default_factory=list)
    completeness: str = Field(default="complete", pattern=r"^(complete|partial)$")
    footing: str = Field(default="none", pattern=r"^(exact|none)$")
    footing_evidence: str | None = None
    members: list[str] = Field(default_factory=list)
    proposed_relation: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _bridge_shape(self) -> BridgeDefinition:
        if self.bridge_type is BridgeType.non_additive_set:
            if len(self.members) < 2 or not self.proposed_relation:
                raise ValueError(f"{self.id}: a non-additive set needs >=2 members and the relation it rejects")
        else:
            if not self.starting_fact or not self.ending_fact:
                raise ValueError(f"{self.id}: additive bridges need starting and ending facts")
            if not self.components and not self.unquantified_components:
                raise ValueError(f"{self.id}: additive bridges need components or named unquantified components")
        return self


class ToleranceResult(StrictModel):
    rounding_floor: Decimal
    cap: Decimal
    allowed: Decimal
    disclosure_unit: Decimal
    footing: str
    input_precisions: dict[str, Decimal]


class ReconciliationResult(StrictModel):
    id: str
    bridge_name: str
    period: str
    scope: str
    bridge_type: BridgeType
    starting_value: Decimal | None = None
    ending_value: Decimal | None = None
    reported_change: Decimal | None = None
    component_fact_ids: dict[str, str] = Field(default_factory=dict)
    identified_components: dict[str, Decimal] = Field(default_factory=dict)
    identified_change: Decimal | None = None
    unquantified_components: list[str] = Field(default_factory=list)
    residual: Decimal | None = None
    tolerance: ToleranceResult | None = None
    status: ReconciliationStatus
    reason: str
    input_fact_ids: list[str] = Field(default_factory=list)
    circular: bool = False
    precision_limited: bool = False

    @model_validator(mode="after")
    def _status_invariants(self) -> ReconciliationResult:
        if self.status is ReconciliationStatus.NOT_APPLICABLE and self.residual is not None:
            raise ValueError(f"{self.id}: NOT_APPLICABLE results cannot carry a residual")
        if self.status is ReconciliationStatus.RECONCILED:
            if self.circular:
                raise ValueError(f"{self.id}: a circular bridge can never be RECONCILED")
            if self.residual is None or self.tolerance is None:
                raise ValueError(f"{self.id}: RECONCILED requires a computed residual and tolerance")
            if abs(self.residual) > self.tolerance.allowed:
                raise ValueError(f"{self.id}: residual exceeds allowed tolerance")
            if self.unquantified_components:
                raise ValueError(f"{self.id}: a bridge with unquantified components cannot be RECONCILED")
        return self


class CalculatedValue(StrictModel):
    fact_id: str
    status: CalcStatus
    value: Decimal | None = None
    uncertainty: Decimal | None = None
    reason: str | None = None


class Severity(str, Enum):
    error = "error"
    warning = "warning"


class ValidationIssue(StrictModel):
    severity: Severity
    code: str
    subject_id: str
    message: str


class ValidationReport(StrictModel):
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.error]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.warning]

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self, severity: Severity | None = None) -> set[str]:
        return {i.code for i in self.issues if severity is None or i.severity is severity}
