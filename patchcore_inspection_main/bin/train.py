import argparse
import contextlib
import logging
import os
import sys

import click
import numpy as np
import torch

# 添加环境变量


import patchcore_inspection_main.src.patchcore as patchcore
import patchcore_inspection_main.src.patchcore.backbones
import patchcore_inspection_main.src.patchcore.common
import patchcore_inspection_main.src.patchcore.metrics
import patchcore_inspection_main.src.patchcore.patchcore
import patchcore_inspection_main.src.patchcore.sampler
import patchcore_inspection_main.src.patchcore.utils

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore_inspection_main.src.patchcore.datasets.mvtec", "MVTecDataset"]}


def main(config=None):
    parser = argparse.ArgumentParser(description='Run PatchCore')
    parser.add_argument('--results_path', type=str, help='Path to store the results')
    parser.add_argument('--gpu', type=int, default=[0], help='GPU indices to use')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--log_group', type=str, default='group', help='Logging group name')
    parser.add_argument('--log_project', type=str, default='project', help='Logging project name')
    parser.add_argument('--save_segmentation_images', type=bool, default=True, help='Flag to save segmentation images')
    parser.add_argument('--save_patchcore_model', type=bool, default=True, help='Flag to save Patchcore model')

    parser.add_argument('--patch_core_config', type=dict, default=None, help='Parameters for patch_core class')
    parser.add_argument('--sampler_config', type=dict, default=None, help='Parameters for sampler_config class')
    parser.add_argument('--dataset_config', type=dict, default=None, help='Parameters for dataset_config class')
    args = parser.parse_args()
    if config is not None:
        print(f"Running experiment with config: {config}")
        for key, value in config.items():
            setattr(args, key, value)
    del config

    run(args)


def run(args):
    print(f"Running experiment with: {args}")
    results_path = args.results_path
    gpu = args.gpu
    seed = args.seed
    log_group = args.log_group
    log_project = args.log_project
    save_segmentation_images = args.save_segmentation_images
    save_patchcore_model = args.save_patchcore_model

    patch_core_config = args.patch_core_config
    sampler_config = args.sampler_config
    dataset_config = args.dataset_config

    the_patch_core = Patch_core(patch_core_config)
    the_sampler = Sampler(sampler_config)
    the_dataset = Dataset(dataset_config)

    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    list_of_dataloaders = the_dataset.get_dataloaders(seed)

    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name

        with device_context:
            torch.cuda.empty_cache()
            imagesize = dataloaders["training"].dataset.imagesize
            sampler = the_sampler.get_sampler(
                device,
            )
            PatchCore_list = the_patch_core.get_patchcore(imagesize, sampler, device)
            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                )
                torch.cuda.empty_cache()
                PatchCore.fit(dataloaders["training"])

            torch.cuda.empty_cache()
            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                scores, segmentations, labels_gt, masks_gt = PatchCore.predict(
                    dataloaders["testing"]
                )
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            # (Optional) Plot example images.
            if save_segmentation_images:
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                def image_transform(image):
                    # 将图片裁剪和resize
                    in_std = np.array(
                        dataloaders["testing"].dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        dataloaders["testing"].dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                image_save_path = os.path.join(
                    run_save_path, "segmentation_images", dataset_name
                )
                os.makedirs(image_save_path, exist_ok=True)
                patchcore.utils.plot_segmentation_images(
                    image_save_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            LOGGER.info("Computing evaluation metrics.")
            auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )["auroc"]

            # Compute PRO score & PW Auroc for all images
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )
            full_pixel_auroc = pixel_scores["auroc"]

            # Compute PRO score & PW Auroc only images with anomalies
            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in sel_idxs],
                [masks_gt[i] for i in sel_idxs],
            )
            anomaly_pixel_auroc = pixel_scores["auroc"]

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

            # (Optional) Store PatchCore model for later re-use.
            # SAVE all patchcores only if mean_threshold is passed?
            if save_patchcore_model:
                patchcore_save_path = os.path.join(
                    run_save_path, "models", dataset_name
                )
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


class Patch_core:
    def __init__(self,
                 patch_core_config
                 ):
        parser = argparse.ArgumentParser(description='Description of your program')

        # Pretraining-specific parameters.
        parser.add_argument("--backbone_names", type=str, default=[])
        parser.add_argument("--layers_to_extract_from", type=str, default=[])
        # Parameters for Glue-code (to merge different parts of the pipeline.
        parser.add_argument("--pretrain_embed_dimension", type=int, default=1024)
        parser.add_argument("--target_embed_dimension", type=int, default=1024)
        parser.add_argument("--preprocessing", default="mean", help='Preprocessing method, mean or conv')
        parser.add_argument("--aggregation", default="mean", help='mean or mlp')
        # Nearest-Neighbour Anomaly Scorer parameters.
        parser.add_argument("--anomaly_scorer_num_nn", type=int, default=5)
        # Patch-parameters.
        parser.add_argument("--patchsize", type=int, default=3)
        parser.add_argument("--patchscore", type=str, default="max")
        parser.add_argument("--patchoverlap", type=float, default=0.0)
        parser.add_argument("--patchsize_aggregate", type=int, default=[])
        # NN on GPU.
        parser.add_argument("--faiss_on_gpu", type=bool, default=True)
        parser.add_argument("--faiss_num_workers", type=int, default=8)

        args = parser.parse_args()
        if patch_core_config is not None:
            for key, value in patch_core_config.items():
                setattr(args, key, value)

        self.backbone_names = args.backbone_names
        self.layers_to_extract_from = args.layers_to_extract_from
        self.pretrain_embed_dimension = args.pretrain_embed_dimension
        self.target_embed_dimension = args.target_embed_dimension
        self.preprocessing = args.preprocessing
        self.aggregation = args.aggregation
        self.patchsize = args.patchsize
        self.patchscore = args.patchscore
        self.patchoverlap = args.patchoverlap
        self.anomaly_scorer_num_nn = args.anomaly_scorer_num_nn
        self.patchsize_aggregate = args.patchsize_aggregate
        self.faiss_on_gpu = args.faiss_on_gpu
        self.faiss_num_workers = args.faiss_num_workers

        backbone_names = list(args.backbone_names)
        if len(backbone_names) > 1:
            self.layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
            for layer in args.layers_to_extract_from:
                idx = int(layer.split(".")[0])
                layer = ".".join(layer.split(".")[1:])
                self.layers_to_extract_from_coll[idx].append(layer)
        else:
            self.layers_to_extract_from_coll = [args.layers_to_extract_from]

    def get_patchcore(self, input_shape, sampler, device):
        loaded_patchcores = []
        for backbone_name, layers_to_extract_from in zip(
                self.backbone_names, self.layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = patchcore.backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            nn_method = patchcore.common.FaissNN(self.faiss_on_gpu, self.faiss_num_workers)

            patchcore_instance = patchcore.patchcore.PatchCore(device)
            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=self.pretrain_embed_dimension,
                target_embed_dimension=self.target_embed_dimension,
                patchsize=self.patchsize,
                featuresampler=sampler,
                anomaly_scorer_num_nn=self.anomaly_scorer_num_nn,
                nn_method=nn_method,
            )
            loaded_patchcores.append(patchcore_instance)
        return loaded_patchcores


class Sampler:
    def __init__(self, sampler_config):
        parser = argparse.ArgumentParser(description='Description of your program')
        parser.add_argument("--name", type=str, default='approx_greedy_coreset')
        parser.add_argument("--percentage", type=float, default=0.1)

        args = parser.parse_args()
        if sampler_config is not None:
            for key, value in sampler_config.items():
                setattr(args, key, value)

        self.name = args.name
        self.percentage = args.percentage

    def get_sampler(self, device):
        if self.name == "identity":
            return patchcore.sampler.IdentitySampler()
        elif self.name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(self.percentage, device)
        elif self.name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(self.percentage, device)


class Dataset:
    def __init__(self,
                 dataset_config):
        parser = argparse.ArgumentParser(description='Description of your program')

        # Arguments
        parser.add_argument("--name", type=str, default='mvtec', help="Description of the name argument")
        parser.add_argument("--data_path", type=str, default='./data', help="Description of the data_path argument")
        # Options
        parser.add_argument("--subdatasets", type=str, default=[], help="Description of the subdatasets option")
        parser.add_argument("--train_val_split", type=float, default=1,
                            help="Description of the train_val_split option")
        parser.add_argument("--batch_size", type=int, default=2, help="Description of the batch_size option")
        parser.add_argument("--num_workers", type=int, default=8, help="Description of the num_workers option")
        parser.add_argument("--resize", type=int, default=256, help="Description of the resize option")
        parser.add_argument("--imagesize", type=int, default=224, help="Description of the imagesize option")
        parser.add_argument("--augment", type=bool, default=False, help="Description of the augment option")

        args = parser.parse_args()

        if dataset_config is not None:
            for key, value in dataset_config.items():
                setattr(args, key, value)
        self.name = args.name
        self.data_path = args.data_path
        self.subdatasets = args.subdatasets
        self.train_val_split = args.train_val_split
        self.batch_size = args.batch_size
        self.resize = args.resize
        self.imagesize = args.imagesize
        self.num_workers = args.num_workers
        self.augment = args.augment

    def get_dataloaders(self, seed):
        dataset_info = _DATASETS[self.name]
        dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])
        dataloaders = []
        for subdataset in self.subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                self.data_path,
                classname=subdataset,
                resize=self.resize,
                train_val_split=self.train_val_split,
                imagesize=self.imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                augment=self.augment,
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                self.data_path,
                classname=subdataset,
                resize=self.resize,
                imagesize=self.imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

            train_dataloader.name = self.name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            if self.train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    self.data_path,
                    classname=subdataset,
                    resize=self.resize,
                    train_val_split=self.train_val_split,
                    imagesize=self.imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        return dataloaders


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path_project = '/home/yukina/Missile_Fault_Detection/project'
    seed = 0
    config = {
        'seed': seed,
        'results_path': os.path.join(path_project, 'patchcore_inspection_main/log', f'seed={seed}'),
        'log_group': 'IM224_WR50_L2-3_P1_D1024-1024_PS-3_AN-1_S0',
        'log_project': 'MVTecAD_Results',
        'save_segmentation_images': False,
        'patch_core_config': {
            'backbone_names': ['resnet50'],
            'layers_to_extract_from': ['layer2', 'layer3'],
            'anomaly_scorer_num_nn': 5,
            'patchsize': 3,
            'faiss_on_gpu': False,
        },
        'sampler_config': {
            'name': 'approx_greedy_coreset',
            'percentage': 1,
        },
        'dataset_config': {
            'name': 'mvtec',
            'data_path': os.path.join(path_project, 'data/mvtec_ad_raw'),
            'subdatasets': ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill',
                            'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
        }
    }

    main(config)
