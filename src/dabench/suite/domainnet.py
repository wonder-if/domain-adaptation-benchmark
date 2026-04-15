"""DomainNet benchmark suites."""

from dabench.task.uda import make_pairwise_uda_suite

DOMAINNET_CLOSED_SET_UDA = make_pairwise_uda_suite(
    dataset="domainnet",
    suite_id="domainnet_closed_set_uda",
    name="DomainNet closed-set UDA",
    domains=("clipart", "infograph", "painting", "quickdraw", "real", "sketch"),
    source_split="train",
    target_split="train",
    eval_split="test",
    metadata={"domains": ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")},
)

SUITES = (DOMAINNET_CLOSED_SET_UDA,)
