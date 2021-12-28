from t5x import utils
import seqio

from ..gins import task

def main():
    ds = utils.get_dataset(
        utils.DatasetConfig(
            "anli_does_it_follow_that_r1_score_eval",
            task_feature_lengths={"inputs": 1024, "targets": 256},
            split="validation",
            batch_size=1,
            shuffle=False,
            seed=None,
            use_cached=True,
            pack=True,
            use_custom_packing_ops=False,
            use_memory_cache=False,
        ),
        0,
        1,
        seqio.PassThroughFeatureConverter
    )

    iter_ds = iter(ds)
    while True:
        sample = next(iter_ds)
        if sample["is_correct"]:
            print("")
            print(sample["inputs_pretokenized"])
            print(f"Expected label: {sample['targets_pretokenized']}")
            input()
    pass

if __name__ == "__main__":
    main()