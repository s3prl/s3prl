import logging
from dataclasses import dataclass

import numpy as np

from s3prl.dataio.encoder.category import CategoryEncoder

from .base import AugmentedDynamicItemDataset, DataPipe

logger = logging.getLogger(__name__)


@dataclass
class BuildMultiClassTagging(DataPipe):
    sample_rate: int = 16000
    feat_frame_shift: int = 160
    intra_or_inter: str = "intra"

    # input
    start_sec_name: str = "start_sec"
    end_sec_name: str = "end_sec"
    segments_name: str = "segments"

    # output
    tag_name: str = "multiclass_tag"
    tag_len_name: str = "tag_len"
    category_name: str = "tag_category"
    all_category_name: str = "all_tag_category"

    def build_label(
        self,
        segments: dict,
        start_sec: int,
        end_sec: int,
        all_tag_category: CategoryEncoder,
    ):
        frame_num = round(
            (end_sec - start_sec) * self.sample_rate / self.feat_frame_shift
        )

        if self.intra_or_inter == "inter":
            category = all_tag_category
        elif self.intra_or_inter == "intra":
            classes_in_this_segment = sorted(segments.keys())
            category = CategoryEncoder(list(classes_in_this_segment))
        else:
            raise ValueError("Only 'inter' or 'intra' is supported")

        T = np.zeros((frame_num, len(category)), dtype=np.int32)
        for class_name, start_ends in segments.items():
            class_idx = category.encode(class_name)
            for seg_start, seg_end in start_ends:
                rel_seg_start = rel_seg_end = None
                if start_sec <= seg_start and seg_start < end_sec:
                    rel_seg_start = seg_start - start_sec
                if start_sec < seg_end and seg_end <= end_sec:
                    rel_seg_end = seg_end - start_sec
                if rel_seg_start is not None or rel_seg_end is not None:
                    rel_seg_start_frame = (
                        round(rel_seg_start * self.sample_rate / self.feat_frame_shift)
                        if rel_seg_start is not None
                        else None
                    )
                    rel_seg_end_frame = (
                        round(rel_seg_end * self.sample_rate / self.feat_frame_shift)
                        if rel_seg_end is not None
                        else None
                    )
                    T[rel_seg_start_frame:rel_seg_end_frame, class_idx] = 1
        return T, frame_num, category

    def forward(
        self, dataset: AugmentedDynamicItemDataset
    ) -> AugmentedDynamicItemDataset:

        if not dataset.has_tool(self.all_category_name):
            logger.warning(
                f"The input dataset does not contain '{self.all_category_name}'. Generate it on-the-fly. "
            )
            with dataset.output_keys_as([self.segments_name]):
                all_classes = set()
                for item in dataset:
                    segments = item[self.segments_name]
                    for class_name in segments.keys():
                        all_classes.add(class_name)
                all_classes = sorted(all_classes)

            dataset.add_tool(self.all_category_name, CategoryEncoder(all_classes))

        dataset.add_dynamic_item(
            self.build_label,
            takes=[
                self.segments_name,
                self.start_sec_name,
                self.end_sec_name,
                self.all_category_name,
            ],
            provides=[
                self.tag_name,
                self.tag_len_name,
                self.category_name,
            ],
        )
        return dataset
