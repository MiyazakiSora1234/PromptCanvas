// Expands one validated form into the requests to send, one image each.
import type { GenerateRequest, Limits } from "../api/types";
import { sizeForAspect, type PickedImage } from "./images";

export interface SeriesItem {
  request: GenerateRequest;
  /** 1-based pose image this request uses (null without pose references). */
  poseNumber: number | null;
}

const SEED_RANGE = 2 ** 32;

/**
 * `count` images per pose image (or just `count` without poses), in that order. A fixed seed
 * continues across all of them (seed, seed+1, ...). With several pose images, each gets its own
 * aspect ratio at the form's pixel count (the form's size already fits the first one).
 */
export function buildSeries(base: GenerateRequest, count: number, poses: PickedImage[], limits: Limits): SeriesItem[] {
  const items: SeriesItem[] = [];
  const groups = poses.length > 0 ? poses : [null];
  groups.forEach((pose, p) => {
    const size =
      pose && p > 0 ? sizeForAspect(pose.width, pose.height, base, limits) : { width: base.width, height: base.height };
    for (let i = 0; i < count; i++) {
      const seed = base.seed === null ? null : (base.seed + items.length) % SEED_RANGE;
      items.push({
        request: { ...base, ...size, seed, num_images: 1, pose_image: pose?.dataUrl ?? null },
        poseNumber: pose ? p + 1 : null,
      });
    }
  });
  return items;
}
