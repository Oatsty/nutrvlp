from typing import Optional
import torchvision.transforms as transforms
from transformers import pipeline, SamProcessor, SamModel
import torch
import torch.nn as nn


class SAM(nn.Module):
    """ SAM 'everything' mode segmentation

    Parameters
    ---
    model_name (str): SAM model checkpoint
    device (torch.device): torch.device
    """
    def __init__(self, model_name: str, device: torch.device):
        super(SAM, self).__init__()
        self.generator = pipeline("mask-generation", model = model_name, device = device, points_per_batch = 256)

    def forward(self, image: torch.Tensor):
        outputs = self.generator(image, points_per_batch = 256)
        return outputs

class SAM2(nn.Module):
    def __init__(self, model_name: str, device: torch.device):
        super(SAM2, self).__init__()
        self.image_processor = SamProcessor.from_pretrained(model_name).image_processor
        self.model = SamModel.from_pretrained(model_name).to(device)
        self.device = device
        self.size = 1024
        self.resize = transforms.Resize(self.size)

    def preprocess(
        self,
        image,
        points_per_batch=64,
        crops_n_layers: int = 0,
        crop_overlap_ratio: float = 512 / 1500,
        points_per_crop: Optional[int] = 32,
        crop_n_points_downscale_factor: Optional[int] = 1,
    ):
        B, C, H, W = image.shape
        target_size = self.image_processor.size["longest_edge"]
        crop_boxes, grid_points, _, input_labels = self.image_processor.generate_crop_boxes(
            image, target_size, crops_n_layers, crop_overlap_ratio, points_per_crop, crop_n_points_downscale_factor, device=self.device
        )
        batched_grid_points = grid_points.repeat(B,1,1,1)
        batched_input_labels = input_labels.repeat(B,1,1)
        # model_inputs = self.image_processor(images=image, return_tensors="pt", do_rescale=False)
        # print(model_inputs['reshaped_input_sizes'])
        image = self.resize(image)

        # return {
        #     "input_points": batched_grid_points.to(self.device),
        #     "input_labels": batched_input_labels.to(self.device),
        #     "input_boxes": crop_boxes.to(self.device),
        #     "pixel_values": image,
        #     "reshaped_input_sizes": [[self.size,self.size] for _ in range(B)],
        #     "original_sizes": [[H,W] for _ in range(B)],
        #     # "is_last": is_last,
        #     # **model_inputs,
        # }

        # # with self.device_placement():
        # #     if self.framework == "pt":
        # #         inference_context = self.get_inference_context()
        # #         with inference_context():
        # #             model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
        # #             image_embeddings = self.model.get_image_embeddings(model_inputs.pop("pixel_values"))
        # #             model_inputs["image_embeddings"] = image_embeddings
        n_points = grid_points.shape[1]
        points_per_batch = points_per_batch if points_per_batch is not None else n_points

        # if points_per_batch <= 0:
        #     raise ValueError(
        #         "Cannot have points_per_batch<=0. Must be >=1 to returned batched outputs. "
        #         "To return all points at once, set points_per_batch to None"
        #     )

        for i in range(0, n_points, points_per_batch):
            batched_points = batched_grid_points[:, i : i + points_per_batch, :, :]
            labels = batched_input_labels[:, i : i + points_per_batch]
            # is_last = i == n_points - points_per_batch
            # yield {
            #     "input_points": batched_points,
            #     "input_labels": labels,
            #     "input_boxes": crop_boxes,
            #     # "is_last": is_last,
            #     **model_inputs,
            # }
            yield {
                "input_points": batched_points,
                "input_labels": labels,
                "input_boxes": crop_boxes,
                "pixel_values": image,
                "reshaped_input_sizes": [[self.size,self.size] for _ in range(B)],
                "original_sizes": [[H,W] for _ in range(B)],
                # "is_last": is_last,
                # **model_inputs,
            }


    def _forward(
        self,
        model_inputs,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        mask_threshold=0,
        stability_score_offset=1,
    ):
        # B, C, H, W = model_inputs['pixel_values'].shape
        input_boxes = model_inputs.pop("input_boxes")
        # original_sizes = [[H,W] for _ in range(B)]
        # is_last = model_inputs.pop("is_last")
        original_sizes = model_inputs.pop("original_sizes")
        # original_size = original_sizes[0]
        reshaped_input_sizes = model_inputs.pop("reshaped_input_sizes")

        model_outputs = self.model(**model_inputs)

        # post processing happens here in order to avoid CPU GPU copies of ALL the masks
        low_resolution_masks = model_outputs["pred_masks"]
        all_masks = self.image_processor.post_process_masks(
            low_resolution_masks, original_sizes, reshaped_input_sizes, mask_threshold, binarize=False
        )
        all_iou_scores = model_outputs["iou_scores"]
        outputs = []
        for i in range(len(all_masks)):
            filtered_masks, filtered_iou_scores, boxes = self.image_processor.filter_masks(
                all_masks[i],
                all_iou_scores[i],
                original_sizes[0],
                input_boxes[0],
                pred_iou_thresh,
                stability_score_thresh,
                mask_threshold,
                stability_score_offset,
            )
            outputs.append(
                {
                    "masks": filtered_masks,
                    # "is_last": is_last,
                    "boxes": boxes,
                    "iou_scores": filtered_iou_scores,

                }
            )
        return outputs
            # yield {
            #     "masks": filtered_masks,
            #     # "is_last": is_last,
            #     "boxes": boxes,
            #     "iou_scores": filtered_iou_scores,
            # }

    def postprocess(
        self,
        model_outputs,
        crops_nms_thresh=0.7,
    ):
        iou_scores = model_outputs.pop("iou_scores")
        masks = model_outputs.pop("masks")
        boxes = model_outputs.pop("boxes")
        output_masks, iou_scores, _, _ = self.image_processor.post_process_for_mask_generation(
            masks, iou_scores, boxes, crops_nms_thresh
        )
        return {"masks": output_masks, "scores": iou_scores}

    def postprocess_one(
        self,
        model_outputs,
        crops_nms_thresh=0.7,
    ):
        all_scores = []
        all_masks = []
        all_boxes = []
        for model_output in model_outputs:
            all_scores.append(model_output.pop("iou_scores"))
            all_masks.extend(model_output.pop("masks"))
            all_boxes.append(model_output.pop("boxes"))

        all_scores = torch.cat(all_scores)
        all_boxes = torch.cat(all_boxes)
        output_masks, iou_scores, rle_mask, bounding_boxes = self.image_processor.post_process_for_mask_generation(
            all_masks, all_scores, all_boxes, crops_nms_thresh
        )
        return {"masks": output_masks, "scores": iou_scores}


    def forward(self, image: torch.Tensor):
        all_outputs = []
        for model_input in self.preprocess(image):
            model_outputs = self._forward(model_input)
            # print(model_outputs[0]['masks'])
            if not len(all_outputs):
                all_outputs = model_outputs
            else:
                all_outputs = [{
                    'masks': output['masks'] + mo['masks'],
                    'iou_scores': torch.cat([output['iou_scores'],mo['iou_scores']]),
                    'boxes': torch.cat([output['boxes'],mo['boxes']]),
                } for output, mo in zip(all_outputs,model_outputs)]

        final_outputs = []
        for output in all_outputs:
            final_outputs.append(self.postprocess(output))

        return final_outputs


    # def compute_keep_masks(
    #     self,
    #     masks,
    #     iou_scores,
    #     pred_iou_thresh=0.88,
    #     stability_score_thresh=0.95,
    #     mask_threshold=0,
    #     stability_score_offset=1,
    # ):
    #     iou_scores = iou_scores.flatten(0, 1)
    #     masks = masks.flatten(0, 1)

    #     if masks.shape[0] != iou_scores.shape[0]:
    #         raise ValueError("masks and iou_scores must have the same batch size.")

    #     if masks.device != iou_scores.device:
    #         iou_scores = iou_scores.to(masks.device)

    #     batch_size = masks.shape[0]

    #     keep_mask = torch.ones(batch_size, dtype=torch.bool, device=masks.device)

    #     if pred_iou_thresh > 0.0:
    #         keep_mask = keep_mask & (iou_scores > pred_iou_thresh)

    #     # compute stability score
    #     if stability_score_thresh > 0.0:
    #         stability_scores = _compute_stability_score_pt(masks, mask_threshold, stability_score_offset)
    #         keep_mask = keep_mask & (stability_scores > stability_score_thresh)

    #     converted_boxes = _batched_mask_to_box(masks)


    #     return keep_mask





    #     # post processing happens here in order to avoid CPU GPU copies of ALL the masks
    #     low_resolution_masks = model_outputs["pred_masks"]
    #     all_masks = self.image_processor.post_process_masks(
    #         low_resolution_masks, original_sizes, reshaped_input_sizes, mask_threshold, binarize=False
    #     )
    #     all_iou_scores = model_outputs["iou_scores"]
        # for i in range(len(all_masks)):
        #     filtered_masks, filtered_iou_scores, boxes = self.image_processor.filter_masks(
        #         all_masks[i],
        #         all_iou_scores[i],
        #         original_sizes[0],
        #         input_boxes[0],
        #         pred_iou_thresh,
        #         stability_score_thresh,
        #         mask_threshold,
        #         stability_score_offset,
        #     )
        #     yield {
        #         "masks": filtered_masks,
        #         # "is_last": is_last,
        #         "boxes": boxes,
        #         "iou_scores": filtered_iou_scores,
        #     }

    # def postprocess(
    #     self,
    #     model_outputs,
    #     crops_nms_thresh=0.7,
    # ):
    #     all_scores = []
    #     all_masks = []
    #     all_boxes = []
    #     for model_output in model_outputs:
    #         all_scores.append(model_output.pop("iou_scores"))
    #         all_masks.extend(model_output.pop("masks"))
    #         all_boxes.append(model_output.pop("boxes"))

    #     all_scores = torch.cat(all_scores)
    #     all_boxes = torch.cat(all_boxes)
    #     output_masks, iou_scores, rle_mask, bounding_boxes = self.image_processor.post_process_for_mask_generation(
    #         all_masks, all_scores, all_boxes, crops_nms_thresh
    #     )
    #     return {"masks": output_masks, "scores": iou_scores}

    # def postprocess_all(
    #     self,
    #     model_outputs,
    #     crops_nms_thresh=0.7,
    # ):
    #     iou_scores = model_outputs.pop("iou_scores")
    #     masks = model_outputs.pop("masks")
    #     boxes = model_outputs.pop("boxes")
    #     output_masks, iou_scores, _, _ = self.image_processor.post_process_for_mask_generation(
    #         masks, iou_scores, boxes, crops_nms_thresh
    #     )
    #     return {"masks": output_masks, "scores": iou_scores}

    # # def forward(self, image: torch.Tensor):
    # #     # outputs = self.generator(image, points_per_batch = 256)
    # #     all_outputs = []
    # #     for model_inputs in self.preprocess(image):
    # #         model_outputs = self._forward(model_inputs)
    # #         all_outputs.append(model_outputs)
    # #     outputs = self.postprocess(all_outputs)
    # #     return outputs

    # # def forward(self, image: torch.Tensor):
    # #     model_inputs = self.preprocess(image)
    # #     model_outputs = self._forward(model_inputs)
    # #     outputs = self.postprocess(model_outputs)
    # #     return outputs

    # def forward(self, image: torch.Tensor):
    #     # outputs = self.generator(image, points_per_batch = 256)
    #     B, _, _, _ = image.shape
    #     all_outputs = [[] for _ in range(B)]
    #     for model_inputs in self.preprocess(image):
    #         for i, model_outputs in enumerate(self._forward(model_inputs)):
    #             all_outputs[i].append(model_outputs)
    #     outputs = []
    #     for i in range(B):
    #         outputs.append(self.postprocess(all_outputs[i]))
    #     return outputs

def _compute_stability_score_pt(masks: torch.Tensor, mask_threshold: float, stability_score_offset: int):
    # One mask is always contained inside the other.
    # Save memory by preventing unnecesary cast to torch.int64
    intersections = (
        (masks > (mask_threshold + stability_score_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    )
    unions = (masks > (mask_threshold - stability_score_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    stability_scores = intersections / unions
    return stability_scores

def _batched_mask_to_box(masks: torch.Tensor):
    """
    Computes the bounding boxes around the given input masks. The bounding boxes are in the XYXY format which
    corresponds the following required indices:
        - LEFT: left hand side of the bounding box
        - TOP: top of the bounding box
        - RIGHT: right of the bounding box
        - BOTTOM: bottom of the bounding box

    Return [0,0,0,0] for an empty mask. For input shape channel_1 x channel_2 x ... x height x width, the output shape
    is channel_1 x channel_2 x ... x 4.

    Args:
        - masks (`torch.Tensor` of shape `(batch, nb_mask, height, width)`)
    """
    # torch.max below raises an error on empty inputs, just skip in this case

    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to Cxheightxwidth
    shape = masks.shape
    height, width = shape[-2:]

    # Get top and bottom edges
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(height, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + height * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(width, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + width * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # Return to original shape
    out = out.reshape(*shape[:-2], 4)
    return out
