from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class COCOAnimalsDataset(BaseSegDataset):
    """COCO-Animals dataset.

    In segmentation map annotation for COCO-Animals, Train-IDs of the 10k version
    are from 1 to 3, where 0 is the ignore index. The ``img_suffix`` is fixed to '.jpg',
    and ``seg_map_suffix`` is fixed to '.png'.
    """
    METAINFO = dict(
        classes=(
            'background', 'cat', 'dog'),
        palette=[[0, 0, 0], [0, 128, 0], [0, 0, 128]])

    def __init__(
        self,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        **kwargs
    ) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
