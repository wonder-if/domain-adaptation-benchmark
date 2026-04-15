"""Office-Home benchmark suites."""

from dabench.task.uda import make_pairwise_uda_suite

OFFICE_HOME_CLOSED_SET_UDA = make_pairwise_uda_suite(
    dataset="office-home",
    suite_id="office_home_closed_set_uda",
    name="Office-Home closed-set UDA",
    domains=("Art", "Clipart", "Product", "Real World"),
    source_split="train",
    target_split="train",
    eval_split="train",
    metadata={"domains": ("Art", "Clipart", "Product", "Real World")},
)

SUITES = (OFFICE_HOME_CLOSED_SET_UDA,)
