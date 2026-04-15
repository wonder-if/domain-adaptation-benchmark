"""Office-31 benchmark suites."""

from dabench.task.uda import make_pairwise_uda_suite

OFFICE31_CLOSED_SET_UDA = make_pairwise_uda_suite(
    dataset="office-31",
    suite_id="office31_closed_set_uda",
    name="Office-31 closed-set UDA",
    domains=("amazon", "dslr", "webcam"),
    source_split="all",
    target_split="all",
    eval_split="all",
    metadata={"domains": ("amazon", "dslr", "webcam")},
)

SUITES = (OFFICE31_CLOSED_SET_UDA,)
